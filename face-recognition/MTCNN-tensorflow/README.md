MTCNN: Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks in TensorFlow   

[论文地址](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)  

模型训练流程:   
1将detection原图进行随机裁剪，与gtbbox iou>0.65为pos, iou>0.4为part, iou<0.3为neg, 统一resize到12，neg无bbox数据, pos和part的bbox数据为gt相对剪裁框的偏移值  
2.landmark原图在bbox附近进行随机裁剪，gtbbox iou>0.65则作为样本 landmark数据为五官gt相对剪裁框的偏移值  
3.训练PNet, 使用neg与pos计算分类loss,pos与part计算bbox loss，landmark独立计算loss  
4.使用scale pyramid将一张detection图片resize为数张不同大小的图片(目的为检测不同大小的目标)，分别输入PNet进行FCN后得到cls与bbox，进行thresh,nms筛选。最后得到不同大小图片筛选过后各自的cls与bbox,将bbox转换为像素值后裁剪出来，统一resize到24  
5.对所有图片重复4，得到所有图片经过PNet后的cls与bbox，与gtbbox进行比对，iou>0.65为pos, iou>0.4为part, iou<0.3为neg,满足多个gtbbox则优选iou最大的  
6.训练RNet,使用neg与pos计算分类loss,pos与part计算bbox loss，landmark独立计算loss   
7.在一张图片上使用4，再输入RNet后得到二次选择的cls与bbox，进行thresh,nms筛选。将bbox转换为像素值后裁剪出来，统一resize到48  
8.对所有图片重复7,得到所有图片经过RNet后的cls与bbox，与gtbbox进行比对，iou>0.65为pos, iou>0.4为part, iou<0.3为neg,满足多个gtbbox则优选iou最大的   
9.训练ONet,使用neg与pos计算分类loss,pos与part计算bbox loss，landmark独立计算loss     

main.py: 入口  


train  
```bash
python main.py --file_name=xxx.jpg
```   


![](demo/example1.jpg)
![](demo/example2.jpg)