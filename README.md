# Regression

## Introduction
**In this blog, we will discuss major important topics that will form a base for *Machine Learning* which are** 
    
    - Simple Linear Regression 
    - Multiple Linear Regression
    - Lasso Regression
    - Ridge Regression
    - Kernel Regression
    - K-Nearest Neighbor
    
    
## What is Regression?

***Regression analysis* is a form of predictive modelling technique which investigates the relationship between a dependent
and an independent variable.**

**The above definition is bookish, in simple terms the regression can be defined as, *Using the relationship between
variables to find the best fit line or the regression equation that can be used to make prediction* **.



<p align="center">
  <img src = "https://miro.medium.com/max/2705/1*KwdVLH5e_P9h8hEzeIPnTg.png"/>
</p>




**There are many types of regressions such as *Linear Regression*, *Polynomial Regression*, *Logistic regression* and
various other *Regression* algorithms but in this blog, we are gonna study *Simple Linear Regression*, *Multiple Linear Regression*, *Lasso Regression*, *Ridge Regression*, *Kernel Regression* & *K-Nearest Neighbor Regression* **.



## Linear Regression



<p align="center">
  <img src = "https://miro.medium.com/max/2400/1*JYeCWrkWtN_iseYlbW79Xw.png"/>
</p>












**Before going into deep, let's first discuss some terminology, examples related to regression so that we can understand it
better later**



**So suppose you want to sell your house and you want to predict the price of your house based on the size of the house then
you can easily predict the price of a house using *Simple Linear Regression*.**




<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-18.png"/>
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-18%20(1).png"/>
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-18%20(2).png"/>
</p>











**So Data will be like**
 





<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-18%20(3).png"/>
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-18%20(4).png"/>
</p>
 





**In the dataset above, we can see there is one input parameter which is the size of the house(*square ft*) while
the output of our *regression* model will be the price of the house(*$$$*).**






## Assumptions in Regression





<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-18%20(5).png"/>
</p>



**From the image above it can be seen that there are some points scatter on the 2D space where *Y-axis* is the *price($$$)* 
of the house and *X-axis* is the *size(sq. ft.)* of the house and points are scattered according to that. Then there is a
green line which we call as the function (f(x)) best fitting on our data(we will see later). But we can see that our function does not pass through all the data points. So if we take a point X<sub> I </sub> marked blue in the image above we
see that there is a gap between the original value and predicted the value at X<sub> I </sub>. This gap is called as the 
error(&epsilon;<sub>i</sub>).**
 
**Similarly, there is some error in all the predicted value of the datapoints.**






<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/CodeCogsEqn.gif"/>
</p>





**We here assume that expected value of &epsilon;<sub>i</sub> is 0 i.e.,**




<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/CodeCogsEqn%20(1).gif"/>
</p>






**By this equation, we meant that it's equally likely that our error is positive or negative. And what does this imply?
This implies that it's equally likely that our observation, the specific observation that we get, is above or below
the functional relationship defined by f(x<sub> I </sub>). So, y<sub>i</sub> is equally likely to be above or below,
f(x<sub>i</sub>).**





<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-18%20(7).png"/>
</p>





**I want to be clear that this is the model that we're using. This is how we're assuming the world works. And there's this
very famous quote by *George Box* that says, *Essentially, all models are wrong, but some are useful*. So what this means
is no models going to be exactly how the world works. It's not going to exactly predict how houses sell, just based on
square feet, or even if you incorporated other things as well. There's always different idiosyncrasies in how the world
works. So above we have discussed the assumption had been taken while designing the algorithm.**



**So we are now gonna discuss the life cycle of a Machine Learning algorithm**




<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-18%20(8).png"/>
</p>





**So every machine learning algorithm start with training data which includes several features/attributes here in our
example of house price prediction there can be many attributes such as house id, house sales, location of a house, 
number of rooms etc... then all those attributes are passed onto next where the feature selection happens i.e. it is
not necessary that all the features in the dataset are good for the prediction of the house price after doing feature
selection via many algorithms(we will discuss later in the blog) we have all the important features of the dataset then
we will pass them through an ML model which will predict the price of a house.**

**So now we have the predicted price of a house but we should have some metrics system which will tell how much close we
have predicted the price or how much accurate our algorithm is. After calculating the errors in the prediction we will
again train our Machine Learning algorithm on the data and repeat the process again and again till we reach to the required
accuracy.**

**So this is the life cycle of Machine Learning algorithm on every dataset.**

**Now we have basic knowledge regarding Machine Learning, let's explore various *Regression algorithms*.**



## Simple Linear Regression

***Simple linear regression* is a *linear regression* model with a single variable as input. That is, it concerns
two-dimensional sample points with one independent variable and one dependent variable (conventionally, the x and y 
coordinates in a Cartesian coordinate system) and finds a linear function (a non-vertical straight line) that, as
accurately as possible, predicts the dependent variable values as a function of the independent variables.
It is the simplest form of all regression algorithms.**






<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-18%20(9).png"/>
</p>





**So going back to our flowchart, what we're gonna talk about now is specifically the machine learning model. So that's
highlighted green box and everything else is grayed out so you can forget about everything else for now. We're just 
talking about our model and what form it takes. So our *Simple Linear Regression* model is just that. It's very simple. 
We're assuming we have just one input, which in this case is, *square feet* of the house and one output which is the house
*sales price* and we're just gonna fit a line. A very simple function here not that quadratic function or higher order 
polynomials we talked about before, just a very simple line. And what's the equation of a line?**
 
**Well, it's just *intercept* plus *slope* times our variable of interest so that we're gonna say that's W<sub>0</sub> + W<sub>1</sub>x. And what this regression model then specifies is that each one of our observations y<sub>i</sub> is
simply that function evaluated at X<sub>i</sub>. So that's  W<sub>0</sub> plus W<sub>i</sub> * X<sub>i</sub> plus the 
error term which we called &epsilon;<sub>i</sub>. So this is our regression model, and to be clear, this error,
&epsilon;<sub>i</sub>, is the distance from our specific observation back down to the line.**




<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-18%20(10).png"/>
</p>





**Equation of Simple Linear Regression is,**






<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/CodeCogsEqn(3).gif"/>
</p>



**While that for any point on the 2D space is, **




<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/CodeCogsEqn(2).gif"/>
</p>






<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-19.png"/>
</p>





**In the equation of Simple Linear Regression we can clearly see that it is nothing but an equation of Straight Line where, slope is w<sub>i</sub> while intercept is w<sub>0</sub>, y<sub>i</sub> is our prediction and x<sub>i</sub> is our input which is the size of the house.
Slope and intercept here are called parameters : regression coefficients. We will explore then in details later in the blog.**




## Fitting line to the data




<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-19%20(1).png"/>
</p>

 
**While working with a Simple Linear Regression model, let's talk about how we're gonna fit a line to data. But before we talk about specific algorithms for fitting, we need to talk about how we're gonna measure the quality of a fit. So, we're gonna talk about this orange box in the above image which is quality metric.**


**So here the metric we are going to use is called Residual Sum of Errors also abbreviated as RSS. It is nothing but sum of squares of difference between original price of house and predicted price of house by our model over whole training dataset i.e.,**




<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-19%20(2).png"/>
</p>




**Above dangerously looking equation can be compressed as below using Sigma(&Sigma;) notation,** 




<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-19%20(3).png"/>
</p>
