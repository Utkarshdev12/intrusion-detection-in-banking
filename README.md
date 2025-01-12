# Intrusion Detection in Banking System using Machine Learning

## Overview
This project focuses on building a machine learning-based Intrusion Detection System (IDS) for the banking sector. It aims to detect various types of attacks such as Denial of Service (DoS), Blackhole, Grayhole, and other network-related anomalies that might compromise the security and performance of banking systems.

The model utilizes machine learning algorithms to analyze network traffic and flag suspicious behavior based on predefined features and attack patterns.

## Features
- **DoS Attacks Detection**: Detects Denial of Service attacks such as flooding and scheduling attacks.
- **Traffic Analysis**: Analyzes network traffic to identify abnormal patterns.
- **Feature Engineering**: Utilizes 19 features related to network behavior for accurate prediction.
- **Machine Learning Algorithms**: Implements various classification algorithms such as Random Forest, Support Vector Machine (SVM), and Decision Trees for attack detection.

## Dataset
The dataset used for this project contains 374,661 instances with 19 features related to the network activities in a banking system. It includes data about:
- **Transaction ID, Time, Is_CH, Rank, Data Sent, Attack Type**, and more.
The dataset is available at Kaggle and can be accessed [here](https://www.kaggle.com/datasets/bassamkasasbeh1/wsnds?select=WSN-DS.csv).

## Installation

### Prerequisites
- Python 3.x
- pip (Python package installer)
- conda
- XAMPP Control Panel

### Setup Instructions
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/Intrusion-Detection-Banking-System.git
2.conda activate base<br>

3 python app.py
