# Regression

## Introduction
In this blog, we will discuss major important topics that will form a base for Machine Learning which are 
    
    - Simple Linear Regression 
    - Multiple Linear Regression
    - Lasso Regression
    - Ridge Regression
    - Kernel Regression
    - K-Nearest Neighbor

## What is Regression?

Regression analysis is a form of predictive modelling technique which investigates the relationship between a dependent
and independent variable.

The above definition is a bookish definition, in simple terms the regression can be defined as, **Using the relationship between
variables to find the best fit line or the regression equation that can be used to make prediction**.


<p align="center">
  <img src = "https://miro.medium.com/max/2705/1*KwdVLH5e_P9h8hEzeIPnTg.png"/>
</p>

There are many types of regressions such as ‘Linear Regression’, ‘Polynomial Regression’, ‘Logistic regression’ and others but
in this blog, we are going to study **Simple Linear Regression**, **Multiple Linear Regression**, **Lasso Regression**, **Ridge Regression**, **Kernel Regression**, **K-Nearest Neighbor Regression**.

## Linear Regression

<p align="center">
  <img src = "https://miro.medium.com/max/2400/1*JYeCWrkWtN_iseYlbW79Xw.png"/>
</p>







### Simple Linear Regression

Simple linear regression is a linear regression model with a single variable as input. That is, it concerns two-dimensional sample points with one independent variable and one dependent variable (conventionally, the x and y coordinates in a Cartesian coordinate system) and finds a linear function (a non-vertical straight line) that, as accurately as possible, predicts the dependent variable values as a function of the independent variables.
It is the simplest form of all regression algorithms.





**Before going into deep, let's first discuss some terms, examples related to regeression**

So suppose you want to sell your house and you want to predict the price of your house on t he basis of size of the house then you can easily predict the price of house using Simple Linear Regression.

<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-18.png"/>
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-18%20(1).png"/>
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-18%20(2).png"/>
</p>








**So here Data will be like**

<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-18%20(3).png"/>
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-18%20(4).png"/>
</p>
 


Here in the dataset we can see there is one input parameter which is the size of the house(**square ft**) while ouptut of our regerssion model will be the price of the house(**$$$**).





### Assumption in Regression


<p align="center">
  <img src = "https://github.com/kampaitees/Linear-Regression/blob/master/Images/2019-12-18%20(5).png"/>
</p>


We can see from the above image that there are some points scatter on the 2D space where Y-axis is the price of the house and X-axis is the size of the house and points are scattered according to that. Then there is green line which we call as the function(f(x)) best fitting on the our data(we will see later). But we can see that our function do not pass through all the data points. So if we take a point Xi(say) marked blue in the image above we see that there is gap between original value and predicted value at Xi. This gap is called as the error($\epsilon\$i).
Similarly there is some error in all the predicted value of the datapoints.

$yi = f(xi) + \epsilon$
