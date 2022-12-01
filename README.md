# TransX-Ray

**Nov 24th, 2022**
>>Nabil Bachroin
1. DenseNet121=*0,67*
2. Swin=*0,61*
>> Jeremie
1. Mx=*0,704*
2. Denstrans=*0,661* [max=93,7 min=0,42]

**Dec 1st, 2022**
>>Nabil Bachroin
1. DenseNet121=*0,701* [max=Edema(0,855) min=Nodule(0,603)]
^^(N-Enhcd, i_shape=(224,224,3), N-Aug, U-Norm, optm=sgd)
^^Pipeline4-Dense.ipynb
2. DenseNet121=*0,7361* [max=Edema(0,897) min=Cardiomegaly(0,625)]
^^(U-Enhcd, i_shape=(320,320,3), U-Aug, U-Norm, optm=adam)
^^Aug=brightness,zoom
^^Pipeline5-Dense-1Des22.ipynb
3. DenseNet121=*0,706* [max=Edema(0,868) min=Nodule(0,636)]
^^(N-Enhcd, i_shape=(320,320,3), U-Aug, U-Norm, optm=adam)
^^Aug=brightness,zoom
^^Pipeline5-Dense-1Des22-NE.ipynb
4. SwinTransformer=*0,637* [max=Edema(0,829) min=Nodule(0,531)]
^^(U-Enhcd, i_shape=(224,224,3), U-Aug, U-Norm, optm=adam)
^^Aug=brightness,zoom
^^Pipeline5-SM-1Des22.ipynb