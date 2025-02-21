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
    <h2 align="center">
        <img src='./result/figure/init_stat.png'>
    </h2>
    
    It should be noted that please download the initial environment files classified in this article first and place them in the corresponding directory:
    ```bash
        Bench4Merge\ 
          DJI_init\
            DJI_high_dhw_results.pkl
            DJI_medium_dhw_results.pkl
            DJI_low_dhw_results.pkl
        # You can choose which initial environment to extract in create_init.py
    ```

    Those files can be obtained at:

  - Second, run the merging process base on the initial environment
    ```bash
        python create_init.py
    ```

    We provide three implemented methods for comparison: RL method, RL combined with MPC method, and optimization based method. You can implement your personal algorithm based on this framework and compare with them.

    The existing model files can be obtained at：

  - Third, observe and analyze the merging process
    The vehicle's trajectory is saved in:
    ```bash
        Bench4Merge\ 
          result\
            trajectory.csv # Trajectory information of the entire process
            trajectory_1.csv # Trajectory information of the last 1 frame
            trajectory_10.csv # Trajectory information of the last 10 frame
    ```
    You can watch a video of the merging process through vis2.ipynb
    <h2 align="center">
        <img src='./result/figure/result_video.png'>
    </h2>


# Comprehensive evaluation based on LLM

This article uses the Alibaba Q-wen 70B large model as the evaluation, and the prompt design and calling method are as follows:

```bash
    python evaluate.py
```






