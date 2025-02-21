# Bench4Merge
A Comprehensive Benchmark for Merging in Realistic Dense Traffic with Micro-Interactive Vehicles

<h2 align="center">
  <img src='./result/figure/overall.jpg'>
</h2>

<h2 align="center">
What can Bench4Merge provide ?<b> Click to view the video.</b>
<br>
<b>&#x2193;&#x2193;&#x2193;</b>
</h2>

[![Bench4Merge]('./result/figure/overall.jpg')](https://youtu.be/2ZBHL5UC4_c?si=Hw3YmFOiFiDbqxZ5 "Bench4Merge")

# Dataset
In order to train main road vehicles in dense traffic environments, we extracted over 500000 following segments that meet the requirements of dense traffic scenarios from three public traffic datasets. Based on this, we trained and obtained the micro interaction behavior of main road vehicles mentioned in this article.

Each data sample contains information about the **Target Vehicle, Leading Vehicle, Interactive Vehicle, and the Map Information**. Each vehicle is represented by an **11-dimensional vector**, which corresponds to the following parameters in order: 

<h2 align="center">
  <img src='./result/figure/data_sample.png'>
</h2>

To achieve the .pkl data, please clink this link:

# Using Process
How to use the Bench4Merge?
  - First, generate an initial environment
    ```bash
        # we make .ipynb for easier observation of the initial environment
        python create_init.py
    ```
    It should be noted that please download the initial environment files classified in this article first and place them in the corresponding directory:
    ```bash
        Bench4Merge\ 
          DJI_init\
            DJI_high_dhw_results.pkl
            DJI_medium_dhw_results.pkl
            DJI_low_dhw_results.pkl
    ```

    Those files can be download at:


