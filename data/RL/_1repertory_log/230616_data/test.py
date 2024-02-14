import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# fig, axs = plt.subplots(2, 2, figsize=(12.8, 10)) # , sharex=True, sharey=True , gridspec_kw={'height_ratios': [1, 1.1], 'width_ratios': [1, 1]}
xlim = (-5.5, 26.5)
ylim = (-5.5, 19.5)
font = fm.FontProperties(fname="C:/Windows/Fonts/simsun.ttc") # times.ttf 


# axs[0,0].set_xlim(xlim)
# axs[0,0].set_ylim(ylim)
# axs[0,0].set_aspect('equal') # Set 1:1 aspect ratio
# axs[0,0].set_xlabel("X")
# axs[0,0].set_ylabel("Y")
# axs[0,0].set_title("设计1", fontproperties=font,fontsize=30)
# axs[0,1].set_xlim(xlim)
# axs[0,1].set_ylim(ylim)
# axs[0,1].set_aspect('equal') # Set 1:1 aspect ratio
# axs[0,1].set_xlabel("X")
# axs[0,1].set_ylabel("Y")
# axs[0,1].set_title("设计2", fontproperties=font,fontsize=30)
# axs[1,0].set_xlim(xlim)
# axs[1,0].set_ylim(ylim)
# axs[1,0].set_aspect('equal') # Set 1:1 aspect ratio
# axs[1,0].set_xlabel("X")
# axs[1,0].set_ylabel("Y")
# axs[1,0].set_title("设计3", fontproperties=font,fontsize=30)
# axs[1,1].set_xlim(xlim)
# axs[1,1].set_ylim(ylim)
# axs[1,1].set_aspect('equal') # Set 1:1 aspect ratio
# axs[1,1].set_xlabel("X")
# axs[1,1].set_ylabel("Y")
# axs[1,1].set_title("设计4", fontproperties=font,fontsize=30)
fig, ax_ = plt.subplots(1, 1, figsize=(12.8, 10)) # , sharex=True, sharey=True , gridspec_kw={'height_ratios': [1, 1.1], 'width_ratios': [1, 1]}
ax_.set_xlim(xlim)
ax_.set_ylim(ylim)
ax_.set_aspect('equal') # Set 1:1 aspect ratio
# ax_.set_xlabel('X')
# ax_.set_ylabel('Y')

ax_.set_xlabel('X', fontsize=19)
ax_.set_ylabel('Y', fontsize=19)
ax_.tick_params(axis='x', labelsize=19)
ax_.tick_params(axis='y', labelsize=19)
ax_.set_title("本方法", fontproperties=font,fontsize=60)

# plt.subplots_adjust(wspace=0.1, hspace=0.25)
plt.tight_layout()

plt.show()