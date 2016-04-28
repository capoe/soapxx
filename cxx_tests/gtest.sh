#! /bin/bash

#soap_tests/test.exe --gtest_list_tests
soap_tests/test.exe
#soap_tests/test.exe --gtest_filter=TestFunctions.GradientYlm
#soap_tests/test.exe --gtest_filter=TestCutoffShiftedCosine.* --gtest_output=xml

