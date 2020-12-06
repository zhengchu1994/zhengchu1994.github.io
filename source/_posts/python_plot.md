---
title: python_plot
mathjax: true
date: 2020-04-08 16:00:00
tags: python & plot
categories: python
visible:

---









### 单图



```python
import numpy as np
import matplotlib.pyplot as plt

# plt.figure(figsize=(80, 40))

# plt.subplot(241)

plt.rcParams['figure.figsize'] = (40, 40) # 设置figure_size尺寸
plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
plt.rcParams['image.cmap'] = 'gray' # 设置 颜色 style

#figsize(12.5, 4) # 设置 figsize

plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
ax = plt.gca() #gca 'get current axes' 获取图像的边框对象==

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
x = [0.01,0.02, 0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
y = [0.5997,0.6728,0.7119,0.7219,0.7317,0.7400,0.7528,0.7618,0.7695,0.7684]
y1=[0.5699,0.6080,0.6410,0.6672,0.6900,0.7077,0.7108,0.7217,0.7284,0.7395]
y2=[0.5578,0.5879,0.6197,0.6454,0.6661,0.6832,0.6954,0.7007,0.7149,0.7205]
y3=[0.6153,0.6430,0.6738,0.6977,0.7139,0.7266,0.7300,0.7434,0.7506,0.7600]
y4=[0.7183,0.7303,0.7533,0.7639,0.7658,0.7714,0.7857,0.7876,0.7888,0.7899]
plt.tick_params(labelsize=35)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Calibri') for label in labels]
[label.set_fontweight('bold') for label in labels]
font1 = {'family' : 'Calibri',
'weight' : 'bold',
'size' : 40,
}
font2 = {'family' : 'Calibri',
'weight' : 'bold',
'size' : 47,
}
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
ax.set_xlim(0,0.11)
ax.set_xticklabels( ('0', '0.02', '0.04', '0.06', '0.08','0.1'))
ax.set_ylim(0.55,0.8)

#ax.set_yticklabels( ('0.55', '0.6', '0.65', '0.7', '0.8'))

plt.plot(x, y, marker='^',label=u'deepwalk',lw=8,ms=14)
plt.plot(x, y1, marker='*',label=u'sdne',lw=8,ms=19)
plt.plot(x, y2, marker='h',label=u'line',lw=8,ms=14)
plt.plot(x, y4, marker='s',label=u'GIC2Gauss',lw=8,ms=14)
plt.plot(x, y3, marker='p',label=u'G2G_oh',lw=8,ms=17)

plt.legend(prop=font1,edgecolor='black',loc = 4) # 让图例生效
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"Percentage of labeled nodes",font1) #X轴标签
plt.ylabel("Micro F1",font1) #Y轴标签

#plt.text(-10,-10,t, wrap=True)

plt.title("Cora",font2) #标题

plt.subplots_adjust(top=0.975,bottom=0.045,left=0.025,right=1,hspace=0.25,wspace=0.15)

#plt.savefig('E:\classification.svg',format='svg',dpi=300)

plt.show()
```

![image-20200408155714523](https://tva1.sinaimg.cn/large/00831rSTgy1gdmeyvm9kjj31gr0u0anx.jpg)









### 8个图放一起



![img](https://tva1.sinaimg.cn/large/00831rSTgy1gdmdb4pnyej32090u04kb.jpg)

```python
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(60, 25))
plt.subplot(241)
ax = plt.gca() #gca ‘get current axes’ 获取图像的边框对象==
== 设置有边框和头部边框颜色为空right、top、bottom、left
ax.spines[‘right’].set_color(‘none’)
ax.spines[‘top’].set_color(‘none’)
x = [0.01,0.02, 0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
y = [0.5997,0.6728,0.7119,0.7219,0.7317,0.7400,0.7528,0.7618,0.7695,0.7684]
y1=[0.5699,0.6080,0.6410,0.6672,0.6900,0.7077,0.7108,0.7217,0.7284,0.7395]
y2=[0.5578,0.5879,0.6197,0.6454,0.6661,0.6832,0.6954,0.7007,0.7149,0.7205]
y3=[0.6153,0.6430,0.6738,0.6977,0.7139,0.7266,0.7300,0.7434,0.7506,0.7600]
y4=[0.7183,0.7303,0.7533,0.7639,0.7658,0.7714,0.7857,0.7876,0.7888,0.7899]
plt.tick_params(labelsize=35)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname(‘Calibri’) for label in labels]
[label.set_fontweight(‘bold’) for label in labels]
font1 = {‘family’ : ‘Calibri’,
‘weight’ : ‘bold’,
‘size’ : 40,
}
font2 = {‘family’ : ‘Calibri’,
‘weight’ : ‘bold’,
‘size’ : 47,
}
plt.rcParams[‘xtick.direction’] = ‘in’
plt.rcParams[‘ytick.direction’] = ‘in’
ax.set_xlim(0,0.11)
ax.set_xticklabels( (‘0’, ‘0.02’, ‘0.04’, ‘0.06’, ‘0.08’,‘0.1’))
ax.set_ylim(0.55,0.8)
#ax.set_yticklabels( (‘0.55’, ‘0.6’, ‘0.65’, ‘0.7’, ‘0.8’))
plt.plot(x, y, marker=’^’,label=u’deepwalk’,lw=8,ms=14)
plt.plot(x, y1, marker=’*’,label=u’sdne’,lw=8,ms=19)
plt.plot(x, y2, marker=‘h’,label=u’line’,lw=8,ms=14)
plt.plot(x, y4, marker=‘s’,label=u’GIC2Gauss’,lw=8,ms=14)
plt.plot(x, y3, marker=‘p’,label=u’G2G_oh’,lw=8,ms=17)

plt.legend(prop=font1,edgecolor=‘black’,loc = 4) # 让图例生效
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"Percentage of labeled nodes",font1) #X轴标签
plt.ylabel(“Micro F1”,font1) #Y轴标签
#plt.text(-10,-10,t, wrap=True)
plt.title(“Cora”,font2) #标题

plt.subplot(242)
ax = plt.gca() #gca ‘get current axes’ 获取图像的边框对象
设置有边框和头部边框颜色为空right、top、bottom、left
ax.spines[‘right’].set_color(‘none’)
ax.spines[‘top’].set_color(‘none’)
y5=[0.5959,0.6447,0.6853,0.6972,0.7118,0.7214,0.7334,0.7436,0.7533,0.7522]
y6=[0.5643,0.5984,0.6319,0.6560,0.6855,0.6919,0.7068,0.7106,0.7187,0.7254]
y7=[0.5527,0.5803,0.6176,0.6384,0.6600,0.6732,0.6854,0.6907,0.7049,0.7135]
y8=[0.6207,0.6395,0.6577,0.6782,0.6971,0.7062,0.7191,0.7234,0.7306,0.7332]
y9=[0.6833,0.7012,0.7339,0.7450,0.7455,0.7529,0.7660,0.7703,0.7689,0.7801]
plt.tick_params(labelsize=35)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname(‘Calibri’) for label in labels]
[label.set_fontweight(‘bold’) for label in labels]
#plt.plot(x, y, ‘ro-’)
#plt.plot(x, y1, ‘bo-’)
#pl.xlim(-1, 11) # 限定横轴的范围
#pl.ylim(-1, 110) # 限定纵轴的范围
#ax.set_xticks([0.5,1,1.5,2,2.5,3])
ax.set_xlim(0,0.11)
ax.set_xticklabels( (‘0’, ‘0.02’, ‘0.04’, ‘0.06’, ‘0.08’,‘0.1’))
ax.set_ylim(0.55,0.8)
#ax.set_yticks([-70,-50,-30,-10,10,30,50,70])
plt.rcParams[‘xtick.direction’] = ‘in’
plt.rcParams[‘ytick.direction’] = ‘in’
plt.plot(x, y5, marker=’^’,label=u’deepwalk’,lw=8,ms=14)
plt.plot(x, y6, marker=’*’,label=u’sdne’,lw=8,ms=19)
plt.plot(x, y7, marker=‘h’,label=u’line’,lw=8,ms=14)
plt.plot(x, y9, marker=‘s’,label=u’GIC2Gauss’,lw=8,ms=14)
plt.plot(x, y8, marker=‘p’,label=u’G2G_oh’,lw=8,ms=17)
plt.legend(prop=font1,edgecolor=‘black’,loc = 4) # 让图例生效
#plt.xticks(x, names)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"Percentage of labeled nodes",font1) #X轴标签
plt.ylabel(“Macro F1”,font1) #Y轴标签
plt.title(“Cora”,font2) #标题

plt.subplot(243)
ax = plt.gca() #gca ‘get current axes’ 获取图像的边框对象
设置有边框和头部边框颜色为空right、top、bottom、left
ax.spines[‘right’].set_color(‘none’)
ax.spines[‘top’].set_color(‘none’)
y10=[0.4267, 0.4887,0.5249,0.5634,0.5768, 0.6012,0.6228,0.6251, 0.6247,0.6204]
y11=[0.4078,0.4591,0.486,0.5031,0.5152,0.5255,0.5341,0.5472,0.5427,0.5523]
y12=[0.3614,0.4153,0.4347,0.4612,0.4725,0.4780,0.4850,0.4931,0.4952,0.5034]
y13=[0.3824,0.4373,0.4624,0.4934,0.5071,0.5120,0.5242,0.5278,0.5303,0.5368]
y14=[0.5382,0.5724,0.5923,0.6103,0.6157,0.6241,0.6292,0.6289,0.6359,0.6421]
plt.tick_params(labelsize=35)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname(‘Calibri’) for label in labels]
[label.set_fontweight(‘bold’) for label in labels]
#plt.plot(x, y, ‘ro-’)
#plt.plot(x, y1, ‘bo-’)
#pl.xlim(-1, 11) # 限定横轴的范围
#pl.ylim(-1, 110) # 限定纵轴的范围
#ax.set_xticks([0.5,1,1.5,2,2.5,3])
ax.set_xlim(0,0.11)
ax.set_xticklabels( (‘0’, ‘0.02’, ‘0.04’, ‘0.06’, ‘0.08’,‘0.1’))
#ax.set_yticks([0.3,0.4,0.6])
ax.set_ylim(0.3,0.65)
#ax.set_yticks([-70,-50,-30,-10,10,30,50,70])
plt.rcParams[‘xtick.direction’] = ‘in’
plt.rcParams[‘ytick.direction’] = ‘in’
plt.plot(x, y10, marker=’^’,label=u’deepwalk’,lw=8,ms=14)
plt.plot(x, y11, marker=’*’,label=u’sdne’,lw=8,ms=19)
plt.plot(x, y12, marker=‘h’,label=u’line’,lw=8,ms=14)
plt.plot(x, y14, marker=‘s’,label=u’GIC2Gauss’,lw=8,ms=14)
plt.plot(x, y13, marker=‘p’,label=u’G2G_oh’,lw=8,ms=17)
plt.legend(prop=font1,edgecolor=‘black’,loc = 4) # 让图例生效
#plt.xticks(x, names)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"Percentage of labeled nodes",font1) #X轴标签
plt.ylabel(“Micro F1”,font1) #Y轴标签
plt.title(“Citeseer”,font2) #标题

plt.subplot(244)
ax = plt.gca() #gca ‘get current axes’ 获取图像的边框对象
== 设置有边框和头部边框颜色为空right、top、bottom、left==
ax.spines[‘right’].set_color(‘none’)
ax.spines[‘top’].set_color(‘none’)
y15=[ 0.3922,0.4651,0.5110,0.5488, 0.5624, 0.5897,0.6122,0.6131,0.6122,0.6110]
y16=[0.3818,0.4238,0.4441,0.4649,0.4831,0.5015,0.5087,0.5211,0.5255,0.5373]
y17=[0.3450,0.3855,0.4130,0.4300,0.4519,0.4640,0.4664,0.4707,0.4789,0.4840]
y18=[0.3708,0.4035,0.4244,0.4462,0.4695,0.4843,0.4980,0.5022,0.5061,0.5146]
y19=[0.5105,0.5545,0.5743,0.5952,0.6012,0.6114,0.6171,0.6166,0.6248,0.6312]
plt.tick_params(labelsize=35)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname(‘Calibri’) for label in labels]
[label.set_fontweight(‘bold’) for label in labels]
#plt.plot(x, y, ‘ro-’)
#plt.plot(x, y1, ‘bo-’)
#pl.xlim(-1, 11) # 限定横轴的范围
#pl.ylim(-1, 110) # 限定纵轴的范围
#ax.set_xticks([0.5,1,1.5,2,2.5,3])
ax.set_xlim(0,0.11)
ax.set_xticklabels( (‘0’, ‘0.02’, ‘0.04’, ‘0.06’, ‘0.08’,‘0.1’))
#ax.set_yticks([0,0.2,0.4,0.6])
ax.set_ylim(0.3,0.65)
#ax.set_yticks([-70,-50,-30,-10,10,30,50,70])
plt.rcParams[‘xtick.direction’] = ‘in’
plt.rcParams[‘ytick.direction’] = ‘in’
plt.plot(x, y15, marker=’^’,label=u’deepwalk’,lw=8,ms=14)
plt.plot(x, y16, marker=’*’,label=u’sdne’,lw=8,ms=19)
plt.plot(x, y17, marker=‘h’,label=u’line’,lw=8,ms=14)
plt.plot(x, y19, marker=‘s’,label=u’GIC2Gauss’,lw=8,ms=14)
plt.plot(x, y18, marker=‘p’,label=u’G2G_oh’,lw=8,ms=17)
plt.legend(prop=font1,edgecolor=‘black’,loc = 4) # 让图例生效
#plt.xticks(x, names)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"Percentage of labeled nodes",font1) #X轴标签
plt.ylabel(“Macro F1”,font1) #Y轴标签
plt.title(“Citeseer”,font2) #标题

plt.subplot(245)
ax = plt.gca() #gca ‘get current axes’ 获取图像的边框对象
设置有边框和头部边框颜色为空right、top、bottom、left
ax.spines[‘right’].set_color(‘none’)
ax.spines[‘top’].set_color(‘none’)
x = [0.01,0.02, 0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
y0 = [0.6992,0.7420, 0.7630, 0.7723, 0.7881, 0.7918,0.7915,0.7941,0.8001,0.8017]
y_1= [0.6929,0.7152,0.7352,0.7412,0.7491,0.7562,0.7695,0.7711,0.7781,0.7801]
y_2=[0.6711,0.6904,0.7083,0.7245,0.7358,0.7475,0.7530,0.7570,0.7627,0.7685]
y_3=[0.6798,0.6983,0.7080,0.7170,0.7233,0.7273,0.7297,0.7320,0.7329,0.7341]
y_4=[0.7845,0.7937,0.7941,0.7956,0.7987,0.7990,0.7999,0.8010,0.8030,0.8040]
plt.tick_params(labelsize=35)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname(‘Calibri’) for label in labels]
[label.set_fontweight(‘bold’) for label in labels]
plt.rcParams[‘xtick.direction’] = ‘in’
plt.rcParams[‘ytick.direction’] = ‘in’
ax.set_xlim(0,0.11)
ax.set_xticklabels( (‘0’, ‘0.02’, ‘0.04’, ‘0.06’, ‘0.08’,‘0.1’))
#ax.set_yticks([0,0.2,0.4,0.6,0.8])
ax.set_ylim(0.55,0.85)
plt.plot(x, y0, marker=’^’,label=u’deepwalk’,lw=8,ms=14)
plt.plot(x, y_1, marker=’*’,label=u’sdne’,lw=8,ms=19)
plt.plot(x, y_2, marker=‘h’,label=u’line’,lw=8,ms=14)
plt.plot(x, y_4, marker=‘s’,label=u’GIC2Gauss’,lw=8,ms=14)
plt.plot(x, y_3, marker=‘p’,label=u’G2G_oh’,lw=8,ms=17)
plt.legend(prop=font1,edgecolor=‘black’,loc = 4,ncol = 2) # 让图例生效
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"Percentage of labeled nodes",font1) #X轴标签
plt.ylabel(“Micro F1”,font1) #Y轴标签
#plt.text(-10,-10,t, wrap=True)
plt.title(“Pubmed”,font2) #标题

plt.subplot(246)
ax = plt.gca() #gca ‘get current axes’ 获取图像的边框对象
设置有边框和头部边框颜色为空right、top、bottom、left
ax.spines[‘right’].set_color(‘none’)
ax.spines[‘top’].set_color(‘none’)
y_5=[0.6774,0.7154,0.7387,0.7520,0.7709,0.7753,0.7753, 0.7784,0.7852,0.7861]
y_6=[0.6659,0.7045,0.7275,0.7340,0.7409,0.7528,0.7529,0.7559,0.7589,0.7629]
y_7=[0.6434,0.6744, 0.6993,0.7093,0.7185,0.7237,0.7362,0.7392,0.7463,0.7538]
y_8=[0.6617,0.6826,0.6929,0.7014,0.7071,0.7118,0.7140,0.7166,0.7177,0.7192]
y_9=[0.7673,0.7782,0.7781,0.7804,0.7824, 0.7827,0.7856,0.7857,0.7850,0.7862]
plt.tick_params(labelsize=35)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname(‘Calibri’) for label in labels]
[label.set_fontweight(‘bold’) for label in labels]
#plt.plot(x, y, ‘ro-’)
#plt.plot(x, y1, ‘bo-’)
#pl.xlim(-1, 11) # 限定横轴的范围
#pl.ylim(-1, 110) # 限定纵轴的范围
#ax.set_xticks([0.5,1,1.5,2,2.5,3])
ax.set_xlim(0,0.11)
ax.set_xticklabels( (‘0’, ‘0.02’, ‘0.04’, ‘0.06’, ‘0.08’,‘0.1’))
#ax.set_yticks([0,0.2,0.4,0.6,0.8])
ax.set_ylim(0.55,0.85)
#ax.set_yticks([-70,-50,-30,-10,10,30,50,70])
plt.rcParams[‘xtick.direction’] = ‘in’
plt.rcParams[‘ytick.direction’] = ‘in’
plt.plot(x, y_5, marker=’^’,label=u’deepwalk’,lw=8,ms=14)
plt.plot(x, y_6, marker=’*’,label=u’sdne’,lw=8,ms=19)
plt.plot(x, y_7, marker=‘h’,label=u’line’,lw=8,ms=14)
plt.plot(x, y_9, marker=‘s’,label=u’GIC2Gauss’,lw=8,ms=14)
plt.plot(x, y_8, marker=‘p’,label=u’G2G_oh’,lw=8,ms=17)
plt.legend(prop=font1,edgecolor=‘black’,ncol = 2,loc = 4) # 让图例生效
#plt.xticks(x, names)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"Percentage of labeled nodes",font1) #X轴标签
plt.ylabel(“Macro F1”,font1) #Y轴标签
plt.title(“Pubmed”,font2) #标题

plt.subplot(247)
ax = plt.gca() #gca ‘get current axes’ 获取图像的边框对象
== 设置有边框和头部边框颜色为空right、top、bottom、left==
ax.spines[‘right’].set_color(‘none’)
ax.spines[‘top’].set_color(‘none’)
y_10=[0.6722,0.7085,0.7181,0.7238,0.7430,0.7457,0.7552,0.7629,0.7712,0.7799]
y_11=[0.6423,0.6716,0.6975,0.7093,0.7181,0.7291,0.7319,0.7391,0.7414,0.7456]
y_12=[0.5989,0.6443,0.6780,0.6883,0.6979,0.7035,0.7131,0.7166,0.7236,0.7292]
y_13=[0.7181,0.7469,0.7610,0.7667,0.7706,0.7750,0.7770,0.7791,0.7818,0.7825]
y_14=[0.7887,0.8001,0.7977,0.8000,0.8046,0.8038,0.8046,0.8019,0.8056,0.8050]
plt.tick_params(labelsize=35)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname(‘Calibri’) for label in labels]
[label.set_fontweight(‘bold’) for label in labels]
#plt.plot(x, y, ‘ro-’)
#plt.plot(x, y1, ‘bo-’)
#pl.xlim(-1, 11) # 限定横轴的范围
#pl.ylim(-1, 110) # 限定纵轴的范围
#ax.set_xticks([0.5,1,1.5,2,2.5,3])
ax.set_xlim(0,0.11)
ax.set_xticklabels( (‘0’, ‘0.02’, ‘0.04’, ‘0.06’, ‘0.08’,‘0.1’))
#ax.set_yticks([0,0.2,0.4,0.6,0.8])
ax.set_ylim(0.5,0.85)
#ax.set_yticks([-70,-50,-30,-10,10,30,50,70])
plt.rcParams[‘xtick.direction’] = ‘in’
plt.rcParams[‘ytick.direction’] = ‘in’
plt.plot(x, y_10, marker=’^’,label=u’deepwalk’,lw=8,ms=14)
plt.plot(x, y_11, marker=’*’,label=u’sdne’,lw=8,ms=19)
plt.plot(x, y_12, marker=‘h’,label=u’line’,lw=8,ms=14)
plt.plot(x, y_14, marker=‘s’,label=u’GIC2Gauss’,lw=8,ms=14)
plt.plot(x, y_13, marker=‘p’,label=u’G2G_oh’,lw=8,ms=17)
plt.legend(prop=font1,edgecolor=‘black’,ncol = 2,loc = 4) # 让图例生效
#plt.xticks(x, names)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"Percentage of labeled nodes",font1) #X轴标签
plt.ylabel(“Micro F1”,font1) #Y轴标签
plt.title(“DBLP”,font2) #标题

plt.subplot(248)
ax = plt.gca() #gca ‘get current axes’ 获取图像的边框对象
设置有边框和头部边框颜色为空right、top、bottom、left
ax.spines[‘right’].set_color(‘none’)
ax.spines[‘top’].set_color(‘none’)
y_15=[0.5924,0.6248,0.6493,0.6657,0.6743,0.6857,0.6997,0.7074,0.7065,0.7128]
y_16=[0.5212,0.5683,0.6002,0.6255,0.6380,0.6518,0.658,0.6621,0.6720,0.6797]
y_17=[0.5577,0.5894,0.6131,0.6383,0.6578,0.6714,0.6793,0.6816,0.6888,0.6910]
y_18=[0.6102,0.6464,0.6779,0.6969, 0.7020,0.7069,0.7102,0.7142,0.7187,0.7194]
y_19=[0.7340,0.7461,0.7424, 0.7459,0.7539,0.7522,0.7522,0.7529,0.7522, 0.7539]
plt.tick_params(labelsize=35)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname(‘Calibri’) for label in labels]
[label.set_fontweight(‘bold’) for label in labels]
#plt.plot(x, y, ‘ro-’)
#plt.plot(x, y1, ‘bo-’)
#pl.xlim(-1, 11) # 限定横轴的范围
#pl.ylim(-1, 110) # 限定纵轴的范围
#ax.set_xticks([0.5,1,1.5,2,2.5,3])
ax.set_xlim(0,0.11)
ax.set_xticklabels( (‘0’, ‘0.02’, ‘0.04’, ‘0.06’, ‘0.08’,‘0.1’))
#ax.set_yticks([0,0.2,0.4,0.6,0.8])
ax.set_ylim(0.5,0.85)
#ax.set_yticks([-70,-50,-30,-10,10,30,50,70])
plt.rcParams[‘xtick.direction’] = ‘in’
plt.rcParams[‘ytick.direction’] = ‘in’
plt.plot(x, y_15, marker=’^’,label=u’deepwalk’,lw=8,ms=14)
plt.plot(x, y_17, marker=’*’,label=u’sdne’,lw=8,ms=19)
plt.plot(x, y_16, marker=‘h’,label=u’line’,lw=8,ms=14)
plt.plot(x, y_19, marker=‘s’,label=u’GIC2Gauss’,lw=8,ms=14)
plt.plot(x, y_18, marker=‘p’,label=u’G2G_oh’,lw=8,ms=17)
plt.legend(prop=font1,edgecolor=‘black’,ncol = 2,loc = 4) # 让图例生效
#plt.xticks(x, names)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"Percentage of labeled nodes",font1) #X轴标签
plt.ylabel(“Macro F1”,font1) #Y轴标签
plt.title(“DBLP”,font2) #标题
plt.tight_layout()
plt.subplots_adjust(top=0.975,bottom=0.045,left=0.025,right=1,hspace=0.25,wspace=0.15)
plt.savefig(‘E:\classification.svg’,format=‘svg’,dpi=300)
plt.show()
```





----



### 饼图

![image-20200408141925991](https://tva1.sinaimg.cn/large/00831rSTgy1gdmc553sxej30qi096ta8.jpg)





```python
import matplotlib.pyplot as plt
import matplotlib

#plt.rcParams['figure.figsize'] = (15, 12) # 设置
font2 = {'family' : 'Calibri',
'weight' : 'bold',
'size' : 18,
}
### graph1
cora_labels = ['Training', 'Validation','Testing','Unused']
cora_size = [20 * 7, 500, 1000, 2708 - 1000 - 500 - 20 * 7]

explode = (0.3,0.01,0.01,0.01)
plt.figure(figsize=(15, 7))
plt.subplot(131)
ax1 = plt.gca()
ax1.pie(cora_size, explode = explode, labels =cora_labels,autopct="%1.1f%%", shadow =False, pctdistance=0.85, startangle=90)
# plt.rcParams['figure.figsize'] = (80, 40) # 设置
#centre_circle = plt.Circle((0,0),0.70,fc='white')
#fig = plt.gcf()
#fig.gca().add_artist(centre_circle)
plt.title("CORA",font2)
ax1.axis('equal')
plt.tight_layout()

### graph2

citeseer_labels = ['Training', 'Validation','Testing','Unused']
citeseer_size = [20 * 6, 500, 1000, 3327 - 1000 - 500 - 20 * 6]

explode = (0.3,0.01,0.01,0.01)

plt.subplot(132)
ax2 = plt.gca()
ax2.pie(citeseer_size, explode = explode, labels =citeseer_labels,autopct="%1.1f%%", shadow =False, pctdistance=0.85, startangle=90)
# plt.rcParams['figure.figsize'] = (80, 40) # 设置
#fig.gca().add_artist(centre_circle)
plt.title("Citeseer",font2)
ax2.axis('equal')
plt.tight_layout()
#plt.xlabel(r"Citeseer") #X轴标签

### graph3
pubmed_labels = ['Training', 'Validation','Testing','Unused']
pubmed_size = [20 * 3, 500, 1000, 19717 - 1000 - 500 - 20 * 3]

explode = (0.3,0.01,0.01,0.01)

plt.subplot(133)
ax3 = plt.gca()
ax3.pie(pubmed_size, explode = explode, labels =pubmed_labels,autopct="%1.1f%%", shadow =False, pctdistance=0.85, startangle=90)
# plt.rcParams['figure.figsize'] = (80, 40) # 设置
#centre_circle = plt.Circle((0,0),0.70,fc='white')
#fig = plt.gcf()
#fig.gca().add_artist(centre_circle)
plt.title("Pubmed",font2)
ax3.axis('equal')
plt.tight_layout()
plt.margins(0)
#plt.subplots_adjust(bottom=0.15)
#plt.xlabel(r"Pubmed") #X轴标签
#plt.margins(0)
#plt.subplots_adjust(bottom=0.6)
plt.savefig('data.pdf',format='pdf',dpi=300)
plt.show()
```

![image-20200408155512792](https://tva1.sinaimg.cn/large/00831rSTgy1gdmewrrvepj31v60u0tf3.jpg)



* 标签在外部：

```python
fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

cora_labels = ['Training', 'Validation','Testing','Unused']
cora_size = [20 * 7, 500, 1000, 2708 - 1000 - 500 - 20 * 7]

wedges, texts = ax.pie(cora_size, wedgeprops=dict(width=0.5), startangle=-40)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(cora_labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, **kw)

ax.set_title("CORA")


plt.show()
```





![image-20200408221129092](https://tva1.sinaimg.cn/large/00831rSTgy1gdmpsbjrihj31gw0u0dkn.jpg)

