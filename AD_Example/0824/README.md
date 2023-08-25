# Multipath, K=6

环境车皆为真实数据；主车前10帧为真实数据，后续交通流为模型仿真，主车位于(0,0处)。基于未来10帧样本数据聚类K轨迹类。损失函数如下：

$$\mathcal{L}(\theta) = - \sum_{m=1}^M \sum_{k=1}^K 1\{k = \hat{k}^m\} \left[ \log\pi (a^k|x^m;\theta) + \sum_{t=1}^T \log \mathcal{N}(s_t^k|a_t^k+\mu_t^k,\Sigma_t^k;\theta) \right].$$

效果如下：

<img src="https://github.com/tianshapojun/INTERACTION_Dataset/assets/10208337/66044ed3-8264-46b5-aec4-3768253600ce" width="450px">
<img src="https://github.com/tianshapojun/INTERACTION_Dataset/assets/10208337/577500f7-2b0a-4a3d-9249-5ad64bc8287d" width="450px">
<img src="https://github.com/tianshapojun/INTERACTION_Dataset/assets/10208337/782b4fae-74d2-4c63-b9a7-111a13be743a" width="450px">
<img src="https://github.com/tianshapojun/INTERACTION_Dataset/assets/10208337/1ffe2353-fdc9-4ada-93c4-da9336015186" width="450px">

# Multipath, K=10, yaw optimized

<img src="https://github.com/tianshapojun/INTERACTION_Dataset/assets/10208337/804e37d6-2ad9-4238-a4bd-3007bab79596" width="450px">
<img src="https://github.com/tianshapojun/INTERACTION_Dataset/assets/10208337/4d969157-1c11-40ba-9980-0eb63180adcc" width="450px">
<img src="https://github.com/tianshapojun/INTERACTION_Dataset/assets/10208337/28f04173-3947-4bf7-8213-dd119fba746a" width="450px">
<img src="https://github.com/tianshapojun/INTERACTION_Dataset/assets/10208337/9b8dda0b-acb6-46c0-bbf9-46c913a8d30e" width="450px">
<img src="https://github.com/tianshapojun/INTERACTION_Dataset/assets/10208337/7b1e027c-49f4-4f16-9c83-aadb507beae5" width="450px">
<img src="https://github.com/tianshapojun/INTERACTION_Dataset/assets/10208337/c37861e8-5064-43a2-8480-fdb075cfd467" width="450px">
