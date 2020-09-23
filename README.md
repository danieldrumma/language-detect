# language-detect

This was a project completed in Spring of 2020 to allow for the determination of text as Dutch or English with use of adaboost and decision trees.
The data used is extracted from wikipedia.
A script for creating data is included.


run with classifyMain

enter command line input as 

a - file to train on   ---another_train.dat
b - training file for test file ...this is to compare the test file once it has recieved predictions from the hypothesis
    trained on data from file a   -----train_.dat
c - test file - used to test hypothesis trained from file a   ----test_.dat

d - algorithm to run ---> "dt" for decision tree or "ada" for adaboost
a         b          c         d
train.dat train_.dat test_.dat dt
or
train.dat train_.dat test.dat ada

Modify max depth and number of stumps with macros in classifyMain
