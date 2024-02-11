# Python Unit Testing


## Overview

Unit testing is a method for testing software which looks at the smallest testable pieces of code, called units, are tested for correct operation.

With unit testing, we are able to verify that each part of the code, including helper functions which may not be exposed to the user, works correctly and as intended.

The idea is that we are independently checking each small piece of our program to ensure that it works.

In contrast, regression and integration testing check that the different parts of the program work well together and as intended.

Unit tests are an important part of regression testing to ensure that the code still functions as expected after we have made changes to the code and helps to ensure code stability.

After making changes to our code, we can run the unit tests we have created previously to ensure that the existing functionality in other parts of the codebase has not been impacted by our changes.

Another benefit of unit tests is that they help to easily isolate errors.

With unit tests, we can analyze the outputs of our unit tests to see if any component of our code has been throwing errors and start debugging from there.

## Test Driven Development

Testing is important to good software development that there’s even a software development process based on testing which is Test Driven Development (TDD).

Three rules of TDD are:

1. You are not allowed to write any production code unless it is to make a failing unit test pass.

2. You are not allowed to write any more of a unit test than is sufficient to fail; and compilation failures are failures.

3. You are not allowed to write any more production code than is sufficient to pass the one failing unit test.

The key idea of TDD is that we base our software development around a set of unit tests that we have created which makes unit testing the heart of the TDD software development process. In this way, you assured that you have a test for every component you developed.

TDD also biased towards having smaller tests which means tests that are more specific and test fewer components at a time which aids in tracking down errors and smaller tests are also easier to read and understand since there are less components at play in a single run.


## PyUnit fFamework

PyUnit is Python’s built-in unit testing framework and is Python’s version of the corresponding JUnit testing framework for Java.

```py
  # Our code to be tested
  class Rectangle:
      def __init__(self, width, height):
          self.width = width
          self.height = height

      def get_area(self):
          return self.width * self.height

      def set_width(self, width):
          self.width = width

      def set_height(self, height):
      self.height = height
```

Here is the complete script for the unit test:

```py
  import unittest

  class TestGetAreaRectangleWithSetUp(unittest.TestCase):

    @classmethod
    def setUpClass(self):
      #this method is only run once for the entire class rather than being run for each test which is done for setUp()
      self.rectangle = Rectangle(0, 0)

    def test_normal_case(self):
      self.rectangle.set_width(2)
      self.rectangle.set_height(3)
      self.assertEqual(self.rectangle.get_area(), 6, "incorrect area")

    def test_geq(self):
      """tests if value is greater than or equal to a particular target"""
      self.assertGreaterEqual(self.rectangle.get_area(), -1)

    def test_assert_raises(self):
      """using assertRaises to detect if an expected error is raised when running a particular block of code"""
      with self.assertRaises(ZeroDivisionError):
        a = 1 / 0
```

## Unit Testing in Action

Now, we teat a function that gets stock data from Yahoo Finance using pandas_datareader and doing this in PyUnit:

```py
  import datetime
  import unittest

  import pandas as pd
  import pandas_datareader.data as web

  def get_stock_data(ticker):
      """pull data from stooq"""
      df = web.DataReader(ticker, 'yahoo')
      return df

  class TestGetStockData(unittest.TestCase):
      @classmethod
      def setUpClass(self):
          """We only want to pull this data once for each TestCase since it is an expensive operation"""
          self.df = get_stock_data('^DJI')

      def test_columns_present(self):
          """ensures that the expected columns are all present"""
          self.assertIn("Open", self.df.columns)
          self.assertIn("High", self.df.columns)
          self.assertIn("Low", self.df.columns)
          self.assertIn("Close", self.df.columns)
          self.assertIn("Volume", self.df.columns)

      def test_non_empty(self):
          """ensures that there is more than one row of data"""
          self.assertNotEqual(len(self.df.index), 0)

      def test_high_low(self):
          """ensure high and low are the highest and lowest in the same row"""
          ohlc = self.df[["Open","High","Low","Close"]]
          highest = ohlc.max(axis=1)
          lowest = ohlc.min(axis=1)
          self.assertTrue(ohlc.le(highest, axis=0).all(axis=None))
          self.assertTrue(ohlc.ge(lowest, axis=0).all(axis=None))

      def test_most_recent_within_week(self):
          """most recent data was collected within the last week"""
          most_recent_date = pd.to_datetime(self.df.index[-1])
          self.assertLessEqual((datetime.datetime.today() - most_recent_date).days, 7)

          unittest.main()
```

Having a unit test framework can help you identify if your data preprocessing is working as expected.

Using unit tests, we are able to identify if there was a material change in the output of our function, and can be a part of a Continuous Integration (CI) process.

We can also attach other unit tests as required depending on the functionality that we depend on from that function.



## References

[A Gentle Introduction to Unit Testing in Python](https://machinelearningmastery.com/a-gentle-introduction-to-unit-testing-in-python/)


