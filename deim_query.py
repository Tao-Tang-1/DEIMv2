import sys
import os
import builtins
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import colorsys

# ==========================================================
# UTF-8 patch
# ==========================================================
_original_open = builtins.open

def _utf8_open(file, mode='r', *args, **kwargs):
    if 'r' in mode and 'encoding' not in kwargs:
        file_str = str(file).lower()
        if file_str.endswith(('.yml', '.yaml', '.json', '.txt')):
            kwargs['encoding'] = 'utf-8'
    return _original_open(file, mode, *args, **kwargs)

builtins.open = _utf8_open

sys.path.insert(0, r"E:\jaas\code\DEIMv2")
from engine.core import YAMLConfig


# =========================
# Monkey-patch decoder to capture per-layer ref_points
# =========================
def patch_decoder_for_query_capture(decoder_module):
    """
    Monkey-patch TransformerDecoder.forward to capture ref_points_detach at each layer.
    Returns a list that will be filled with per-layer ref_points during forward().
    """
    captured = {'ref_points_per_layer': []}
    original_forward = decoder_module.forward

    def patched_forward(target, ref_points_unact, memory, spatial_shapes,
                        bbox_head, score_head, query_pos_head, pre_bbox_head,
                        integral, up, reg_scale, attn_mask=None, memory_mask=None, dn_meta=None):
        from engine.deim.dfine_utils import distance2bbox, weighting_function
        output = target
        output_detach = pred_corners_undetach = 0
        value = decoder_module.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        if not hasattr(decoder_module, 'project'):
            project = weighting_function(decoder_module.reg_max, up, reg_scale)
        else:
            project = decoder_module.project

        ref_points_detach = F.sigmoid(ref_points_unact)
        captured['ref_points_per_layer'] = [ref_points_detach.detach().cpu()]
        query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

        for i, layer in enumerate(decoder_module.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)

            if i >= decoder_module.eval_idx + 1 and decoder_module.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=decoder_module.layer_scale)
                value = decoder_module.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)

            if i == 0:
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + torch.log(
                    ref_points_detach / (1 - ref_points_detach + 1e-6) + 1e-6))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), decoder_module.reg_scale)

            if decoder_module.training or i == decoder_module.eval_idx:
                scores = score_head[i](output)
                scores = decoder_module.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)
                if not decoder_module.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()
            captured['ref_points_per_layer'].append(ref_points_detach.detach().cpu())

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), \
               torch.stack(dec_out_pred_corners), torch.stack(dec_out_refs), pre_bboxes, pre_scores

    decoder_module.forward = patched_forward
    return captured


# =========================
# Visualization 1: Final Top-K detection boxes
# =========================
def draw_topk_boxes(image_pil, boxes, scores, labels, topk=30, output_path=None):
    """Draw Top-K detection boxes on image, colored by confidence."""
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size

    # Sort by score descending, take top-K
    sorted_idx = scores.argsort(descending=True)[:topk]

    font_path = r"C:\Users\15199\Downloads\DejaVuSans-Bold.ttf"
    font_size = max(20, min(W, H) // 50)
    if os.path.exists(font_path):
        font = ImageFont.truetype(font_path, size=font_size)
    else:
        font = ImageFont.load_default()

    line_width = max(2, min(W, H) // 300)

    for rank, idx in enumerate(sorted_idx):
        x1, y1, x2, y2 = boxes[idx].cpu().numpy().astype(int)
        conf = scores[idx].item()

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        # Color: red (high) -> blue (low) gradient
        r = int(255 * conf)
        b = int(255 * (1 - conf))
        color = (r, 0, b)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        label = f"Q{rank} {conf:.3f}"
        text_y = max(0, y1 - font_size - 5)
        draw.text((x1, text_y), label, fill=color, font=font)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        result_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imencode('.jpg', result_cv)[1].tofile(output_path)
        print(f"[OK] Top-K boxes saved: {output_path}")

    return img


# =========================
# Visualization 2: Query refinement trajectories
# =========================
def draw_query_trajectories(image_pil, ref_points_per_layer, scores, topk=20, output_path=None):
    """
    Draw how queries move across decoder layers.
    ref_points_per_layer: list of [1, N, 4] tensors (cxcywh, normalized 0~1)
    """
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size

    # Use final layer scores to select top-K
    final_ref = ref_points_per_layer[-1][0]  # [N, 4]
    sorted_idx = scores.argsort(descending=True)[:topk]

    num_layers = len(ref_points_per_layer)
    print(f"  Trajectory: {topk} queries across {num_layers} layers")

    # Generate distinct colors for each query using HSV
    colors = []
    for i in range(topk):
        hue = i / max(topk, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))

    dot_radius = max(4, min(W, H) // 150)
    line_width = max(2, min(W, H) // 400)

    for rank, q_idx in enumerate(sorted_idx):
        color = colors[rank]
        points = []

        for layer_idx in range(num_layers):
            ref = ref_points_per_layer[layer_idx][0, q_idx]  # [4] cxcywh
            cx, cy = ref[0].item() * W, ref[1].item() * H
            points.append((cx, cy))

        # Draw connecting lines
        for k in range(len(points) - 1):
            x1, y1 = int(points[k][0]), int(points[k][1])
            x2, y2 = int(points[k + 1][0]), int(points[k + 1][1])
            draw.line([x1, y1, x2, y2], fill=color, width=line_width)

        # Draw dots at each layer position
        for k, (cx, cy) in enumerate(points):
            ix, iy = int(cx), int(cy)
            # Layer 0 = large dot, final = large dot, intermediate = smaller
            r = dot_radius * (1.5 if k == 0 or k == len(points) - 1 else 0.8)
            draw.ellipse([ix - r, iy - r, ix + r, iy + r], fill=color)

        # Label at final position
        final_x, final_y = int(points[-1][0]), int(points[-1][1])
        conf = scores[q_idx].item()
        label = f"Q{rank} L{num_layers-1}"

        font_size = max(16, min(W, H) // 60)
        font_path = r"C:\Users\15199\Downloads\DejaVuSans-Bold.ttf"
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, size=font_size)
        else:
            font = ImageFont.load_default()
        draw.text((final_x + dot_radius + 2, final_y - font_size // 2), label, fill=color, font=font)

    # Legend: layer info
    legend_y = 10
    for layer_idx in range(num_layers):
        gray = int(80 + 175 * layer_idx / max(num_layers - 1, 1))
        draw.text((10, legend_y), f"L{layer_idx}", fill=(gray, gray, gray))
        legend_y += font_size + 2

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        result_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imencode('.jpg', result_cv)[1].tofile(output_path)
        print(f"[OK] Query trajectories saved: {output_path}")

    return img


# =========================
# Main
# =========================
def main():
    config_path = r"E:\jaas\code\DEIMv2\configs\deimv2\ablation_experiments\deimv2_dinov3_s_offtype.yml"
    resume_path = r"E:\jaas\code\DEIMv2\outputs\ablation_experiments\deimv2_dinov3_s_offtype_CG_AFS_SCD_QS_MAL_ASL5\best_stg2.pth"

    img_path = r"E:\毕业材料\大论文\数据集\OffType\split\test\images\BIGNAN_2024_LITERAL_BLE_157402_17IVB_Session 2024-06-13 08-38-12_l1_rgb_images_Plot525_Camera1_1.jpg"

    output_dir = "./output/deimv2_query_vis"
    topk = 30

    print("[INFO] Loading model...")
    cfg = YAMLConfig(config_path, resume=resume_path)
    model = cfg.model.eval().cuda()

    checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    model.load_state_dict(state)

    eval_size = cfg.yaml_cfg["eval_spatial_size"]
    postprocessor = cfg.postprocessor.eval().cuda()

    # Patch decoder to capture query positions
    decoder_module = model.decoder.decoder
    captured = patch_decoder_for_query_capture(decoder_module)

    # Load and preprocess image
    img_data = np.fromfile(img_path, dtype=np.uint8)
    orig_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    orig_h, orig_w = orig_img.shape[:2]
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    print(f"[INFO] Image: {orig_w}x{orig_h}, eval_size: {eval_size}")

    transform = T.Compose([T.Resize(eval_size), T.ToTensor()])
    im_tensor = transform(img_pil).unsqueeze(0).cuda()
    orig_size = torch.tensor([[orig_w, orig_h]], device='cuda')

    # Forward pass
    print("[INFO] Running inference...")
    with torch.no_grad():
        output = model(im_tensor)

    # Post-process
    result = postprocessor(output, orig_size)[0]
    boxes = result['boxes']     # [N, 4] xyxy pixel coords
    scores = result['scores']   # [N]
    labels = result['labels']   # [N]

    ref_points_per_layer = captured['ref_points_per_layer']
    num_layers = len(ref_points_per_layer)
    num_queries = ref_points_per_layer[0].shape[1]
    print(f"[INFO] Captured {num_layers} layers, {num_queries} queries")

    # Filter to valid detections for score reference
    valid = scores > 0.1
    print(f"[INFO] Detections with score > 0.1: {valid.sum().item()}")

    # Visualization 1: Top-K boxes
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    box_path = os.path.join(output_dir, f"{img_name}_topk{topk}_boxes.jpg")
    draw_topk_boxes(img_pil, boxes, scores, labels, topk=topk, output_path=box_path)

    # Visualization 2: Query trajectories
    traj_path = os.path.join(output_dir, f"{img_name}_topk{topk}_trajectories.jpg")
    draw_query_trajectories(img_pil, ref_points_per_layer, scores, topk=topk, output_path=traj_path)

    print(f"\n[DONE] Results in: {output_dir}")


if __name__ == '__main__':
    main()
