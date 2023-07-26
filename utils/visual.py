import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

def raster_visual_fig(config, box_img, map_img_1, map_img_2):
    plt.subplot(2,3,1)
    plt.imshow(box_img[...,config["history_num_frames"]])
    plt.subplot(2,3,2)
    plt.imshow(map_img_2)
    plt.subplot(2,3,3)
    plt.imshow(box_img[...,config["history_num_frames"]] + map_img_1[:,:,2])
    for i in range(3):
        plt.subplot(2,3,i+4)    
        plt.imshow(map_img_1[:,:,i])
    plt.show()
    
def raster_visual_anim(config, box_img, map_img_1, save_flg = False):
    def update(n):
        fig.clear() 
        plt.imshow(box_img[...,n] + map_img_1[:,:,2]) 

    N = config["history_num_frames"] + 1 
    fig = plt.figure()
    ani = FuncAnimation(fig, update, frames=range(N), interval=100)
    if save_flg:
        print('Begin saving gif')
        ani.save('D:/Code/INTERACTION_Dataset/test.gif', writer='imagemagick', fps=60)
    plt.show()
