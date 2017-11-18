# Supervised SVM classification 
*Note that the multi-class is obtained from several bi-class SVMs, using a one-versus-one (ovo) strategy.*  
* svm_tools.py: Utilities for inspecting SVM data.*  
*plot_tools.py : Utilities for plotting SVC in 2D.*

1. rbf (gaussien) kernal with C=100 and sigma = 0.75  (Iris dataset)

  risk = 20.67%, Empirical risk = 0.18   
  intro1: applying the SVM and plotting the result  
  intro2: inspecting the internal seperator and predicting function of ovo  
  intro3: plotting the internal decision functions

2.  A simple bi-class example  (make_blob+linear kernal) 


(if you want to plot the dataset you build)  
                 X, y = sklearn.datasets.make_blobs(n_samples=100, centers=[[-.5, 1], [1, -.5]], cluster_std=.25)
                pyplot.scatter(X[:,0],X[:,1],c=y);
                pyplot.show()

   Try a nu-svc with 10% for the proportion of the supporting vectors  
   ![SVM01](https://github.com/ftZHOU/readmePicture/blob/master/SVC01.png)  
   
   Try a gaussien kernal and measure the performance with cross-validation
   risk = 14.00%, Empirical risk = 0.12  
   With 10 times more example -> risk = 11.80%, risk = 0.116
3. Digit Recognition   
   digits-001.py: This shows how to get and display the digits.  
   digits-002.py: The use of a multiclass SVM for digits classification.  
   ![SVM02](https://github.com/ftZHOU/readmePicture/blob/master/SVM02.png)  

   if you want to blur an image  
                import cv2
                digit         = inputs[0]
                img           = digit.reshape((28,28))/255.0    
                blurred_digit = cv2.GaussianBlur(digit.reshape((28,28)), (9,9), 0).reshape(28*28)
                blurred_img   = blurred_digit.reshape((28,28))/255.0






# *Reference link 
1. [*Anderson's Iris data set*](https://zh.wikipedia.org/wiki/%E5%AE%89%E5%BE%B7%E6%A3%AE%E9%B8%A2%E5%B0%BE%E8%8A%B1%E5%8D%89%E6%95%B0%E6%8D%AE%E9%9B%86)  
2. [*make_blobs dataset*](http://datahref.com/archives/191)




