from PIL import Image
from pylab import *
from numpy import *
from scipy.ndimage import filters
from scipy.ndimage import measurements,morphology
from scipy.misc import imsave
from PIL import ImageDraw
import os
#创建图片
#img=Image.new("RGBA",(80,40),(10,255,100))
#draw=ImageDraw.Draw(img)
#draw.text((0,0),"hellow world!")
#img.show()
#img=img.convert("RGB")
#img.save("2.jpg")

#显示图片
#im=Image.open("D:/1.jpg")
#im.show()

#图片裁剪
#img=Image.open("D:/1.jpg")
#box=(50,50,500,500)
#region=img.crop(box)
#region.show()

#查看图片信息
#img=Image.open("D:/1.jpg")
#print("图片格式")
#print(img.format)
#print("图片尺寸")
#print(img.size)
#print("图片颜色模式")
#print(img.mode)

#图片旋转
#img=Image.open("D:/2.jpg")
#out=img.rotate(45)#逆时针旋转45度
#out=img.transpose(Image.ROTATE_180)#旋转180度 可以更改度数
#out.show()

#图片调整尺寸
#img=Image.open("D:/2.jpg")
#out=img.resize((128,128))
#out.show()

#图片左右对换
#img=Image.open("D:/1.jpg")
#out=img.transpose(Image.FLIP_LEFT_RIGHT)#flip翻转
#out.show()

#图片类型转换
#img=Image.open("D:/1.jpg")
#img=img.convert("RGB")
#img.show()

#图片缩略图
#img=Image.open("D:/1.jpg")
#img.thumbnail((128,128))
#img.show()

#图片复制和粘贴图像区域
#img=Image.open("D:/2.jpg")
#box=(100,100,400,400)
#region=img.crop(box)
#region=region.transpose(Image.ROTATE_90)
#img.paste(region,box)
#img.show()



#绘制图像，点和线
#img=Image.open("D:/2.jpg")
#im=array(img)#读取图像到数组
#imshow(im)#绘制图像
#x=[100,400,400,100,100]#绘制一些点
#y=[200,200,500,500,200]
#plot(x,y,'r*')#使用红色星状标记绘制点
##plot(x,y,'b*')#使用蓝色星状标记绘制点
##plot(x,y,'g*')#使用绿色星状标记绘制点
##plot(x,y,'c*')#使用青色星状标记绘制点
##plot(x,y,'m*')#使用品红色星状标记绘制点
##plot(x,y,'y*')#使用黄色星状标记绘制点
##plot(x,y,'k*')#使用黑色星状标记绘制点    . 点 o 圆圈 s 正方形 * 星行 + 加号 x 叉号
##plot(x,y,'w*')#使用白色星状标记绘制点
#plot(x[:5],y[:5])#绘制连接前两个点的线
#title('Plotting:"empire.jpg"')
#axis("off")
#show()

#更改图片格式  错误  png无法转换成jpg
#imgs=Image.open("D:/3.png")
#img="D:/3.png"
#outfile = os.path.splitext(img)[0]+'.jpg'
#print(outfile)
#imgs.save(outfile)

#遍历文件 改格式
#root="D:/xxx"
#filelist=os.listdir(root)
#for infile in filelist:
#    infile="D:/xxx/"+infile
#    print(infile)
#    outfile=os.path.splitext(infile)[0]+"x"+".jpg"
#    print(outfile)
#    if infile!=outfile:
#        try:
#            Image.open(infile).save(outfile)
#        except IOError:
#            print("cannot convert"),infile



##图像轮廓和直方图
#im=array(Image.open("D:/2.jpg").convert("L"))#读取图像到数组
#figure()#新建一个图像
#gray()#不使用颜色信息
#axis("equal")
#axis("off")
#contour(im,origin="image")#在原点的左上角显示轮廓图像
#figure()
#hist(im.flatten(),1024)#flatten() 方法将任意数组按照行优先准则转换成一维数组。
#show()

#交互式标注
#im=array(Image.open("D:/2.jpg"))
#imshow(im)
#print("Please click 3 points")
#x=ginput(3)
#print("you click :",x)
#show()

#图像数组表示
#im=array(Image.open("D:/2.jpg"))
#print(im)
#print(im.shape,im.dtype)# 行 列 颜色通道 数据类型
#im[20:500,20:500]=im[40:520,40:520]#可以错位
#im[:,:,0]=0#修改通道值
#imshow(im)
#show()

#im =array(Image.open("D:/1.jpg").convert("L"),"f")# f 将数据转为浮点型 灰度图没有颜色
#print(im.shape,im.dtype)

#灰度变换
#im=array(Image.open("D:/1.jpg"))
#im2=255-im#对图像进行反相处理
#im3=(100.0/255.0)*im+100#将图像像素值变换到100-200区间
#im4=255.0*(im/255.0)**2 #对图像像素值求平方后得到的图像,对比度升高
#figure()
#subplot(2,2,1)
#imshow(im)
#subplot(2,2,2)
#imshow(im2)
#subplot(2,2,3)
#imshow(im3)
#subplot(2,2,4)
#imshow(im4)
#show()


#figure() 建立一个窗口  从这里开始到下一个figure出现或者show出现属于本窗口
#figure() 建立一个窗口
#show()

#图像缩放
#im=array(Image.open("D:/2.jpg"))
#pil_im=Image.fromarray(uint8(im))#将图像数组转化为对象
#out=pil_im.resize((128,128))
#out.show()

#对一幅灰度图进行直方图均衡化  
#def histeq(im,nbr_bins=256):
#    """对一幅灰度图像进行直方图均衡化"""
#    #计算图像的直方图
#    imhist,bins=histogram(im.flatten(),nbr_bins,normed=True)#histogram返回一个元组(频数，分箱的边界)
#    cdf =imhist.cumsum()#cumsum 对于一维输入a,就是当前列之前的和加到当前列上
#    cdf=255*cdf/cdf[-1]#归一化  cdf[-1]取最后一个元素
#    #使用累积分布函数的线性插值，计算新的像素值
#    im2=interp(im.flatten(),bins[:-1],cdf)
#    return im2.reshape(im.shape),cdf#返回直方均衡化后的图像以及用来做像素值映射的累积分分布函数
#im=array(Image.open("D:/5xx.jpg").convert("L"))
#im2,cdf=histeq(im)
#figure()#新建一个图像
#subplot(2,2,1)
#imshow(im)
#subplot(2,2,2)
#imshow(im2)
#subplot(2,2,3)
#hist(im.flatten(),128)
#subplot(2,2,4)
#hist(im2.flatten(),128)
#show()


#图像平均   
#def computer_average(imlist):
#    """计算图像列表的平均图像"""
#    #打开第一幅图像，将其储存在浮点型数组中
#    imlist[0]="D:/xxxx/"+imlist[0]
#    averageim=array(Image.open(imlist[0]),"f")
#    for imname in imlist[1:]:
#        imname="D:/xxxx/"+ imname
#        try:
#            averageim+=array(Image.open(imname))
#        except:
#            print(imname+"....skipped")
#    averageim/=len(imlist)
#    #返回uint8类型的平均图像
#    return array(averageim,"uint8")
#root="D:/xxxx"
#imlist=os.listdir(root)
#outfile=computer_average(imlist)
#imshow(outfile)
#show()


#主成分分析  
#def pca(X):
#    """主成分分析"""
#    #输入：矩阵X，其中该矩阵中储存训练数据，每一行为一条训练数据
#    #返回：投影矩阵（按照维度的重要性排序），方差和均值
#    #获取维数
#    num_data,dim=X.shape #shape功能是读取矩阵的长度 读出来是几乘几的矩阵
#    #数据中心化
#    mean_X=X.mean(axis=0)
#    X=X-mean_X
#    if dim>num_data:
#        #PAC-使用紧致技巧
#        M=dot(X,X.T)#协方差矩阵 dot 矩阵相乘 X.T转制
#        e,EV=linalg.eigh(M)#特征值和特征向量
#        tmp=dot(X.T,EV).T  #这就是紧致技巧
#        V=tmp[::-1]#由于最后的特征向量是我们所需要的，所以需要将其逆转   [::-1]取从后向前（相反）的元素
#        S=sqrt(e)[::-1]#由于特征值是按照递增顺序排列的，所以需要将其逆转  sqrt() 方法返回数字x的平方根
#        for i in range(V.shape[1]):
#            V[:,i]/=S
#    else:
#        #PCA-使用SVD方法
#        U,S,V=linalg.svd(X)
#        V=V[:num_data]#仅仅返回前num_data维的数据才合理
#        #返回投影矩阵，方差和均值
#    return V,S,mean_X
#root="./img/"
#imlist=os.listdir(root)
#im =array(Image.open(root+imlist[0]))#打开一幅图像
#m,n=im.shape[0:2]#获取图片大小
#immbr=len(imlist)#获取图像数目
##创建矩阵，保存所有压平后的图像数据
#immatrix=array([array(Image.open(root+im)).flatten()
#               for im in imlist],"f")
##执行PCA操作
#V,S,immean=pca(immatrix)
##显示一些图像（均值图像和前7个模式）
#figure()
#gray()
#subplot(2,4,1)
#imshow(immean.reshape(72,72))
#for i in range(7):
#    subplot(2,4,i+2)
#    imshow(V[i].reshape(72,72))
#show()

#图像模糊
#im =array(Image.open("D:/2.jpg").convert("L"))
#im2=filters.gaussian_filter(im,5)#最后参数为标准差
#figure()
#subplot(2,2,1)
#imshow(im)
#subplot(2,2,4)
#imshow(im2)
#show()
#图像模糊2 修改颜色通道
#im =array(Image.open("D:/2.jpg"))
#im2=zeros(im.shape)
#for i in range(3):
#    im2[:,:,i]=filters.gaussian_filter(im[:,:,i],5)
#im2=uint8(im2)#in2=array(im2,"uint8")
#figure()
#subplot(2,2,1)
#imshow(im)
#subplot(2,2,4)
#imshow(im2)
#show()


#图像导数
#im =array(Image.open("D:/2.jpg").convert("L"))
##Sobel导数滤波器
#imx=zeros(im.shape)
#filters.sobel(im,1,imx)#第二个参数表示选择x或者y方向导数，第三个参数保存输出的变量
#imy=zeros(im.shape)
#filters.sobel(im,0,imy)
#magnitude=sqrt(imx**2+imy**2)
#figure()
#subplot(2,2,1)
#imshow(im)
#subplot(2,2,2)
#imshow(magnitude)
#subplot(2,2,3)
#imshow(imx)
#subplot(2,2,4)
#imshow(imy)
#show()


#图像导数2
#im =array(Image.open("D:/2.jpg").convert("L"))
#sigma=5#标准差
#imx=zeros(im.shape)
#filters.gaussian_filter(im,(sigma,sigma),(0,1),imx)#第二个参数为使用的标准差
#imy=zeros(im.shape)
#filters.gaussian_filter(im,(sigma,sigma),(1,0),imy)
#magnitude=sqrt(imx**2+imy**2)
#figure()
#subplot(2,2,1)
#imshow(im)
#subplot(2,2,2)
#imshow(magnitude)
#subplot(2,2,3)
#imshow(imx)
#subplot(2,2,4)
#imshow(imy)

#show()


#形态学：对象计数
#载入图像，然后使用阀值化操作，以保证处理图像为二值图像
#im =array(Image.open("D:/2.jpg").convert("L"))
#im=1*(im<128)
#labels,nbr_objects=measurements.label(im)
#print("number of object:",nbr_objects)
##形态学开操作更好地分离各个对象
#im =array(Image.open("D:/2.jpg").convert("L"))
#im_open=morphology.binary_opening(im,ones((9,5)),iterations=2)
#labels_open,nbr_objects_open=measurements.label(im_open)
#print("number of object :",nbr_objects_open)

#图像降噪
#def denoise(im,U_init,tolerance=0.1,tau=0.125,tv_weight=100):
#    """使用A.chambolle(2005)在公式（11）中的计算步骤实现Rudin_Osher_Fatemi(ROF)
#    去噪模型
#    输入：含有噪声的输入图像（灰度图像），U的初始值，TV正则项权值，步长，停业条件
#    输出：去噪和去除纹理后的图像，纹理残留"""
#    m,n=im.shape#噪声图像的大小
#    #初始化
#    U=U_init
#    Px=im#对偶域的x分量
#    Py=im#对偶域的y分量
#    error=1
#    while(error>tolerance):
#        Uold=U
#        #原始变量的梯度
#        GradUx=roll(U,-1,axis=1)-U#变量U梯度的x分量
#        GradUy=roll(U,-1,axis=0)-U#变量U梯度的y分量
#        #跟新对偶变量
#        PxNew=Px+(tau/tv_weight)*GradUx
#        PyNew=Py+(tau/tv_weight)*GradUy
#        NormNew=maximum(1,sqrt(PxNew**2+PyNew**2))

#        Px=PxNew/NormNew#更新x分量(对偶)
#        Py=PyNew/NormNew#更新x分量(对偶)

#        RxPx=roll(Px,1,axis=1)#对x分量进行向右x轴平移
#        RyPy=roll(Py,1,axis=0)#对y分量进行向右y轴平移

#        DivP=(Px-RxPx)+(Py-RyPy)#对偶域的散度
#        U=im+tv_weight*DivP#更新原始变量
#        #更新误差
#        error=linalg.norm(U-Uold)/sqrt(n*m);
#    return U,im-U#去噪后的图像和纹理残留
#im=zeros((500,500))
#im[100:400,100:400]=128
#im[200:300,200:300]=255
#im=im+30*random.standard_normal((500,500))

#U,T=denoise(im,im)
#G=filters.gaussian_filter(im,10)
#figure()
#subplot(2,2,1)
#imshow(im)
#subplot(2,2,2)
#imshow(U)
#subplot(2,2,3)
#imshow(G)
#show()
#im=array(Image.open("D:/2.jpg").convert("L"))
#U,T=denoise(im,im)
#figure()
#subplot(1,2,1)
#imshow(im)
#subplot(1,2,2)
#imshow(U)
#show()