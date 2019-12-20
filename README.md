# Regression

## Introduction
**In this blog series, we will discuss major important topics that will form a base for *Machine Learning* which are** 
    
    - Simple Linear Regression 
    - Multiple Linear Regression
    - Lasso Regression
    - Ridge Regression
    - Kernel Regression
    - K-Nearest Neighbor
    
    
**So particularly in this blog, we are gonna discuss *Simple Linear Regression*, *Optimization algorithm*, *Residual
Sum of Squares* & how to minimize it.**


## What is Regression?

***Regression analysis* is a form of predictive modelling technique which investigates the relationship between a dependent
and an independent variable.**

**The above definition is bookish, in simple terms the regression can be defined as, *Using the relationship between
variables to find the best fit line or the regression equation that can be used to make prediction***.



<p align="center">
  <img src = "https://miro.medium.com/max/2705/1*KwdVLH5e_P9h8hEzeIPnTg.png"/>
</p>




**There are many types of regressions such as *Linear Regression*, *Polynomial Regression*, *Logistic regression* and
various other *Regression* algorithms but in this blog, we are gonna study *Simple Linear Regression*, *Multiple Linear Regression*, *Lasso Regression*, *Ridge Regression*, *Kernel Regression* & *K-Nearest Neighbor Regression***.



## Linear Regression



<p align="center">
  <img src = "https://miro.medium.com/max/2400/1*JYeCWrkWtN_iseYlbW79Xw.png"/>
</p>












**Before going into deep, let's first discuss some terminology, examples related to regression so that we can understand it
better later.**



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
green line which we call as the function (f(x)) best fitting on our data(we will see later).**
 
**But we can see that our function does not pass through all the data points. So if we take a point X<sub> I </sub> marked
blue in the image above we see that there is a gap between the original value and predicted the value at X<sub> I </sub>. 
This gap is called as the error(&epsilon;<sub>i</sub>).**
 
**Similarly, there is some error in all the predicted value of the data points.**






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



**So we are now gonna discuss the life cycle of a Machine Learning algorithm.**




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





**So going back to our flowchart, what we're gonna talk about now is specifically the machine learning model. So
that's highlighted green box and everything else is greyed out so you can forget about everything else for now. We're
just talking about our model and what form it takes. So our *Simple Linear Regression* model is just that. It's very
simple. We're assuming we have just one input, which in this case is, *square feet* of the house and one output which
is the house *sales price* and we're just gonna fit a line. A very simple function here not that quadratic function or
higher-order polynomials we talked about before, just a very simple line. And what's the equation of a line?**
 
**Well, it's just *intercept* plus *slope* times our variable of interest so that we're gonna say that's W<sub>0</sub>
+ W<sub>1</sub> * x.**

**And what this regression model then specifies is that each one of our observations y<sub>i</sub> is
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



**While that for any point on the 2D space is,**




<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/CodeCogsEqn(2).gif"/>
</p>






<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-19.png"/>
</p>





**In the equation of Simple Linear Regression, we can see that it is nothing but an equation of Straight Line
where, slope is w<sub>i</sub> while intercept is w<sub>0</sub>, y<sub>i</sub> is our prediction and x<sub>i</sub> is
our input which is the size of the house. Slope and intercept here are called parameters: regression coefficients.
We will explore then in details later in the blog.**




## Fitting line to the data




<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-19%20(1).png"/>
</p>

 
**While working with a *Simple Linear Regression* model, let's talk about how we're gonna fit a line to data. But
before we talk about specific algorithms for fitting, we need to talk about how we're gonna measure the quality of
a fit. So, we're gonna talk about this orange box in the above image which is quality metric.**


**So here the metric we are going to use is called *Residual Sum of Errors* also abbreviated as *RSS*. It is nothing
but the sum of squares of the difference between the original price of the house and predicted the price of the house by
our model over the wholextraining dataset i.e.,**




<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-19%20(2).png"/>
</p>




**Above dangerously looking equation can be compressed as below using Sigma(*&Sigma;*) notation,** 




<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-19%20(3).png"/>
</p>





**Above equation can also be written as,**


<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/CodeCogsEqn(4).gif"/>
</p>


**Where y<sub>hat</sub> is our prediction at each data point and we calculate it's difference from the original
value, then we are squaring it and doing the sum of all the squares at each data points to calculate the overall
cost.**


**Now we have overall cost of a particular *Regression* model, we can try the different possible combination of 
parameters i.e., w<sub>0</sub> and w<sub>1</sub> which will give different cost and we will take that line which
had minimum cost.**




<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-19%20(4).png"/>
</p>


**So now the questions arise till when we keep on searching the values for parameters and how we will get to know
whether the parameters we got are one having the minimum cost on our dataset i.e. best parameters for the dataset.**

**So to solve this problem there comes the concept of optimization algorithms which will tell us that when to stop
and provide us with the best parameters for the dataset. There are several optimization techniques like *Gradient Descent*,
*Coordinate Descent*, *Adam*, *RMP prop*, *Momentum* but we are gonna discuss only *Gradient Descent*  and *Coordinate
descent* in the blog.**


**Before discussing the optimization algorithm we will first discuss the interpretation of the parameters in the equation
of the *Simple Linear Regression***


## Use & Interpretation of the fitted Line




<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-19%20(6).png"/>
</p>

 
**The above Regression model can be used by the seller as well as by the buyers both. Here it is how.**



<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-19%20(7).png"/>
</p>




**Seller can put the size of his/her house in the equation and can easily predict the house price which he/she wanted 
to sell.**



<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-19%20(8).png"/>
</p>



**Similarly, a buyer can put his/her budget in the equation and can know how much bigger house he/she can buy in their
budget.** 





<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-19%20(9).png"/>
</p>



### Interpreting the coefficients

**W<sub>0</sub> is nothing but it's the price of the house when the size of the house is 0 which doesn't make sense
but W<sub>0</sub> doesn't have any meaning full interpretation.**




<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-19%20(10).png"/>
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-19%20(12).png"/>
</p>




**W<sub>1</sub> is the slope of the line which here means that it is predicted the change in the price of the house(*output*)
per unit change in the size of the house(*input*).**




<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-19%20(11).png"/>
</p>






<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-19%20(12a).png"/>
</p>



**We have seen the metrics(*RSS*) which we will be using to decide whether parameters of a particular model are good enough for the dataset & to decide the best model for a dataset. Let now see the algorithms for fitting the model.**





## Algorithm for fitting the model



<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-19%20(13).png"/>
</p>




**Now that we have an understanding of what the fitted line is, and how we can use it, let's talk about an algorithm
for searching out the space of all possible lines that we might use, and finding the one that best fits the data.
So in particular what we're going to be doing is focusing in on this machine learning algorithm, which is this dark
grey square is shown in the above flow chart.**
 
**So recall that our cost was to find us this *Residual Sum of Squares*, and for any given line, we can compute
the cost of that line. So, for example, we showed three different lines and three different residual sums of squares
here, but our goal was to minimize over all possible W0 and W1 slopes and intercepts, but a question is, how are we
going to do this? So that's the key question that we're looking to address in this section.**




<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-19%20(14).png"/>
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-20.png"/>
</p>




**Let's formalize this idea a little bit more. So here, what we're showing is our *Residual Sum of Squares* and what
we see is it's a function of two variables, w<sub>0</sub> and w<sub>1</sub>. So we can write it generically, let's
just write it as some function g of a variable w<sub>0</sub> and a variable w<sub>1</sub>. And what I've done is I've
gone ahead and plotted the residual sum of squares versus w<sub>0</sub> and w<sub>1</sub> for the data set.** 

**So here along *X-axis* is w<sub>0</sub> and along  *Y-axis* is w<sub>1</sub>. And then we are plotting here our 
*Residual Sum of Squares*. And the blue mesh surface in the image is our *Residual Sum of Squares* for any given
w<sub>0</sub>, w<sub>1</sub> pair. And our objective here is to minimize over all possible combinations of w<sub>0</sub>,
and w<sub>1</sub>.**

**So we want to find the specific value of w<sub>0</sub>. So, we'll call that w<sub>0</sub> hat & w<sub>1</sub> hat,
that minimize this *Residual Sum of Squares*. So, this is our objective. This is an optimization problem, where
specifically the optimization objective is to minimize a function, in this case, of two parameters, two different variables**

**So now we are gonna discuss some optimization techniques which will give us the best parameters, w<sub>0</sub> hat
& w<sub>1</sub> hat which minimizes the *Residual Sum of Squares* thereby providing us with the best model, but before that
we have to discuss some terminologies like what is *Concavity*, *Convexity* and other related terms.**



## An aside on Optimization


## Concave/Convex functions


**Given a function *g(w)* if we construct a line starting at point *(a, g(a))*  and ending at point *(b, g(b))* then the 
line always lie below the function *g(w)* then this type of functions are known as *Concave* functions.**

**And if, given a function *g(w)* if we construct a line starting at point *(a, g(a))*  and ending at point *(b, g(b))* 
then if line always lie above the function *g(w)* then this type of functions are known as *Convex* functions.**


**Last but not the least, given a function *g(w)* if we construct a line starting at point *(a, g(a))*  and ending at 
point *(b, g(b))* then if line sometimes lie above the function & sometimes lie below the function then this type of
functions are neither *Convex* nor *Concave*.**

**Below are examples of each case.**


<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-20%20(1).png"/>
</p>




**Now we know what is *Concavity* and *Convexity*, let's discuss how to find maximum and minimum of a function because
our cost which is *Residual Sum of Squares* is *Convex* and we have to find the parameters that minimize our cost on 
the dataset.**





<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-20%20(2a).png"/>
</p>




**In case of *Concave* function we can only find maximum value while in case of *Convex* functions we can only find a
minimum value.**

**To find the minimum and maximum values in *Convex* and *Concave* functions respectively we have to take the first
derivative of the function and make the derivative of the function as zero and calculate the value where that derivative
is zero. The value where a derivative is zero will give the minimum and maximum value in *Convex* and *Concave* respectively.**

**In case of neither *Concave* nor *Convex* function there can be multiple maximum and minimum values.**

**Let's look at the example below.**


<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-20%20(2).png"/>
</p>





