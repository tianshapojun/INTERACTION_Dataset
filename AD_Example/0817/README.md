# ViT + loss_T + loss_availabel

环境车皆为真实数据；主车前10帧为真实数据，后续交通流为模型仿真，主车位于(0,0处)。基于光栅化特征的ViT算法，修改损失函数为，可以看到车辆控制在冲出可行域的过程中有一个回调的过程：

$$L_{total} = L_{base}(x,y,w) + \lambda_T \cdot L_T(X_{predict}^T,X_{True}^T) + \lambda_a \cdot L_{collision}IOU(box,region_{available}).$$

<img src="https://github.com/tianshapojun/INTERACTION_Dataset/assets/10208337/649fc0fd-639b-4b8a-848b-a2ad0513d388" width="450px">
<img src="https://github.com/tianshapojun/INTERACTION_Dataset/assets/10208337/82070d59-9d67-4229-9606-2c216bddc0df" width="450px">
<img src="https://github.com/tianshapojun/INTERACTION_Dataset/assets/10208337/b9b35c1f-3814-4c3f-a198-26ad1883a4fb" width="450px">
<img src="https://github.com/tianshapojun/INTERACTION_Dataset/assets/10208337/01a7a01e-d245-4089-8446-f839a7393076" width="450px">
<img src="https://github.com/tianshapojun/INTERACTION_Dataset/assets/10208337/75162b07-834b-447e-8894-dc4a210b2f15" width="450px">

