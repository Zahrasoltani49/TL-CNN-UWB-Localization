# UWB Indoor Positioning Dataset Using DWM3001CDK

## Overview

This repository contains an Ultra-Wideband (UWB) ranging dataset collected using a pair of **Qorvo DWM3001CDK** development platforms. The measurements were captured in four different propagation scenarios inside a library and an adjacent meeting room at Tyndall National Institute.

The library and meeting room are connected by a wooden door, enabling data collection under both Line-of-Sight (LOS) and various Non-Line-of-Sight (NLOS) conditions.

---

## Dataset Description

The dataset consists of four different propagation scenarios:

### 1. Line-of-Sight (LOS)

The transmitter and receiver communicate directly without any obstacles between them.

### 2. Human Body

A person stands between the transmitter and receiver, introducing attenuation and multipath effects caused by the human body.

### 3. Wooden Door

The transmitter is placed inside the library while the receiver is located in the adjacent meeting room. The two devices are separated by a closed wooden door.

### 4. Bookshelves + Wall

The transmitter is placed inside the library while the receiver is located in the meeting room. The UWB signal propagates through bookshelves and a wall, representing a challenging NLOS environment.

---

## Data Collection

For each scenario:

* **20 measurement points** were selected.
* **500 ranging measurements** were recorded at each point.
* Measurements were collected using a pair of **Qorvo DWM3001CDK** devices.

Overall, the dataset contains:

* **4 scenarios**
* **20 measurement points per scenario**
* **500 measurements per point**
* **10,000 measurements per scenario**
* **40,000 total measurements**

---

## Dataset Structure

```text
Dataset/
│
├── LOS/
│   ├── Point01.txt
│   ├── Point02.txt
│   └── ...
│
├── Human_Body/
│   ├── Point01.txt
│   ├── Point02.txt
│   └── ...
│
├── Wooden_Door/
│   ├── Point01.txt
│   ├── Point02.txt
│   └── ...
│
├── Bookshelves_Wall/
│   ├── Point01.txt
│   ├── Point02.txt
│   └── ...
│
├── WholeData.txt
└── WholeData.json
```

Each scenario folder contains **20 text files**, where each file corresponds to one measurement point and contains **500 ranging measurements**.

For convenience, all measurements from the four scenarios are also combined into a single dataset named **WholeData**, which is provided in both:

* `WholeData.txt`
* `WholeData.json`

The unified dataset simplifies data loading for machine learning, statistical analysis, and benchmarking.

---

## Data Format

Each measurement record contains the following fields:

| Column             | Description                                                                                          |
| ------------------ | ---------------------------------------------------------------------------------------------------- |
| **true_range**     | Ground-truth distance (in meters) between the transmitter and receiver.                              |
| **measured_range** | Distance estimated by the DWM3001CDK UWB module.                                                     |
| **range_error**    | Difference between the measured range and the ground-truth distance (`measured_range - true_range`). |
| **measurement**    | Sequential measurement index for the corresponding measurement point.                                |
| **anchor_id**      | Identifier of the anchor (receiver) used during data collection.                                     |
| **cir**            | Channel Impulse Response (CIR) samples recorded for the corresponding ranging measurement.           |

---

## Applications

## Applications

This dataset was specifically collected to support **transfer learning**. It is intended for fine-tuning machine learning models pretrained on datasets collected using the **Decawave DWM1000** platform, enabling their adaptation to measurements acquired with the **Qorvo DWM3001CDK** platform.

The dataset is particularly useful in scenarios where collecting a large, labeled dataset for a new UWB platform is **time-consuming and labor-intensive**. By leveraging transfer learning, pretrained models can be efficiently adapted to the target platform using a relatively small amount of new data, thereby reducing data collection effort while maintaining high classification performance.

---

## Citation

If you use this dataset in your research, please cite the associated publication:


---

## License

Please refer to the repository license for the terms governing the use and redistribution of this dataset.
