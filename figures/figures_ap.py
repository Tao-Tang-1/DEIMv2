import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# =====================================
# Data
# =====================================

data = {
    'Model':[
        'YOLOv5m',
        'YOLOv8m',
        'YOLOv10m',
        'YOLOv11m',
        'YOLOv12m',
        'YOLOv26m',
        'RT-DETR',
        'RT-DETRv2',
        'RT-DETRv3',
        'RT-DETRv4',
        'DEIMv1',
        'DEIMv2',
        'DEIM-CSM'
    ],

    'Params':[25.07,25.86,21.23,20.05,20.14,21.77,
              20.0,20.0,22.24,10.37,10.17,9.66,9.66],

    'FLOPs':[64.4,79.1,68.1,68.2,67.7,74.7,
             60.0,60.0,60.0,24.81,24.81,22.15,22.15],

    'AP50':[50.0,55.6,52.1,55.7,57.3,53.5,
            49.5,48.1,52.0,55.4,53.2,60.3,65.5]
}

df = pd.DataFrame(data)

# =====================================
# CVPR Style Colors
# =====================================

YOLO_COLOR = "#4E79A7"
RTDETR_COLOR = "#59A14F"
DEIM_COLOR = "#E15759"

colors = []

for m in df["Model"]:
    if "YOLO" in m:
        colors.append(YOLO_COLOR)
    elif "DEIM" in m:
        colors.append(DEIM_COLOR)
    else:
        colors.append(RTDETR_COLOR)

# =====================================
# Bubble Size
# =====================================

bubble_size = df["FLOPs"] * 10

# =====================================
# Figure
# =====================================

fig, ax = plt.subplots(figsize=(10,7))

# =====================================
# Scatter
# =====================================

for i,row in df.iterrows():

    if row["Model"] == "DEIM-CSM":
        continue

    elif row["Model"] == "DEIMv2":

        ax.scatter(
            row["Params"],
            row["AP50"],
            s=bubble_size[i],
            marker='s',
            color=DEIM_COLOR,
            edgecolors='black',
            linewidth=1.2,
            zorder=6
        )

    else:

        ax.scatter(
            row["Params"],
            row["AP50"],
            s=bubble_size[i],
            color=colors[i],
            edgecolors='black',
            linewidth=0.8,
            alpha=0.85
        )

# =====================================
# Highlight DEIM-CSM
# =====================================

best = df[df["Model"]=="DEIM-CSM"].iloc[0]

ax.scatter(
    best["Params"],
    best["AP50"],
    s=650,
    marker='*',
    color=DEIM_COLOR,
    edgecolors='black',
    linewidth=1.5,
    zorder=20
)

# =====================================
# Label Offsets
# =====================================

offsets = {
    'YOLOv5m':(-1.2,-0.8),
    'YOLOv8m':(-1.4,0.4),
    'YOLOv10m':(-1.5,-0.6),
    'YOLOv11m':(-1.5,0.5),
    'YOLOv12m':(0.2,0.6),
    'YOLOv26m':(0.2,-0.8),

    'RT-DETR':(-1.8,-0.2),
    'RT-DETRv2':(-2.0,-0.8),
    'RT-DETRv3':(0.2,-0.7),
    'RT-DETRv4':(0.2,0.5),

    'DEIMv1':(0.2,-0.8),
    'DEIMv2':(0.2,0.6),
    'DEIM-CSM':(0.2,0.8)
}

# =====================================
# Labels
# =====================================

for _, row in df.iterrows():

    dx, dy = offsets[row["Model"]]

    if row["Model"] == "DEIM-CSM":

        ax.text(
            row["Params"] + dx,
            row["AP50"] + dy,
            row["Model"],
            fontsize=12,
            fontweight='bold',
            color=DEIM_COLOR
        )

    elif row["Model"] == "DEIMv2":

        ax.text(
            row["Params"] + dx,
            row["AP50"] + dy,
            row["Model"],
            fontsize=11,
            fontweight='bold',
            color=DEIM_COLOR
        )

    else:

        ax.text(
            row["Params"] + dx,
            row["AP50"] + dy,
            row["Model"],
            fontsize=8.5
        )

# =====================================
# Axis
# =====================================

ax.set_xlabel('Parameters (M)', fontsize=15)
ax.set_ylabel(r'$AP_{50}$ (%)', fontsize=15)

ax.set_xlim(8,28)
ax.set_ylim(46,68)

ax.grid(
    linestyle='--',
    linewidth=0.6,
    alpha=0.35
)

# =====================================
# Combined Legend
# =====================================

handles = []

for flops in [25, 50, 75]:
    handles.append(ax.scatter([], [], s=flops*10, color='gray', alpha=0.5, label=f'{flops} GFLOPs'))

handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=YOLO_COLOR, markersize=10, label='YOLO Series'))
handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=RTDETR_COLOR, markersize=10, label='RT-DETR Series'))
handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=DEIM_COLOR, markersize=10, label='DEIM Series'))

ax.legend(handles=handles, loc='upper right', frameon=False, fontsize=11, bbox_to_anchor=(0.98, 0.98), borderpad=1.2, labelspacing=1.5)

# =====================================
# Layout
# =====================================

plt.tight_layout()
plt.show()