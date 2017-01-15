import random
from math import exp
import numpy as np

def computeTransferFnctn(summedNeuronInput, alpha):
    activation = 1.0 / (1.0 + exp(-alpha*summedNeuronInput)) 
    return activation   
    
# Compute derivative of transfer function
def computeTransferFnctnDeriv(NeuronOutput, alpha):
    return alpha*NeuronOutput*(1.0 -NeuronOutput)     

def obtainNeuralNetworkSizeSpecs ():
    numInputNodes = 2
    numHiddenNodes = 2
    numOutputNodes = 2   
    print ' '
    print 'This network is set up to run the X-OR problem.'
    print 'The numbers of nodes in the input, hidden, and output layers have been set to 2 each.' 
            
    # We create a list containing the crucial SIZES for the connection weight arrays                
    arraySizeList = (numInputNodes, numHiddenNodes, numOutputNodes)
    
    # We return this list to the calling procedure, 'main'.       
    return (arraySizeList)  
    
def InitializeWeight ():
    randomNum = random.random()
    weight=1-2*randomNum
    return (weight)  
    
    
def initializeWeightArray (weightArraySizeList):
    numBottomNodes = weightArraySizeList[0]
    numUpperNodes = weightArraySizeList[1]
            
    # Initialize the weight variables with random weights
    wt00=InitializeWeight ()
    wt01=InitializeWeight ()
    wt10=InitializeWeight ()
    wt11=InitializeWeight ()    
    weightArray=np.array([[wt00,wt01],[wt10,wt11]])
    return (weightArray)  
    
def obtainRandomXORTrainingValues ():    
    trainingDataSetNum = random.randint(1, 4)
    if trainingDataSetNum >1.1: # The selection is for training lists between 2 & 4
        if trainingDataSetNum > 2.1: # The selection is for training lists between 3 & 4
            if trainingDataSetNum > 3.1: # The selection is for training list 4
                trainingDataList = (1,1,1,0,0) # training data list 4 selected
            else: trainingDataList = (1,0,0,1,1) # training data list 3 selected   
        else: trainingDataList = (0,1,0,1,2) # training data list 2 selected     
    else: trainingDataList = (0,0,1,0,3) # training data list 1 selected 
           
    return (trainingDataList)
    
def computeSingleNeuronActivation(wt0, wt1, input0, input1):
    
    summedNeuronInput = wt0*input0+wt1*input1    
    # Define the scaling parameter alpha to be 1
    alpha = 1.0
    # Pass the summedNeuronActivation and the transfer function parameter alpha into the transfer function
    activation = computeTransferFnctn(summedNeuronInput, alpha)
    return activation
    
def ComputeSingleFeedforwardPass (inputDataList, wWeightArray, vWeightArray):
    input0 = inputDataList[0]
    input1 = inputDataList[1]      
    # Assign the input-to-hidden weights to specific variables
    wWt00 = wWeightArray[0,0]
    wWt01 = wWeightArray[0,1]
    wWt10 = wWeightArray[1,0]       
    wWt11 = wWeightArray[1,1]    
    # Assign the hidden-to-output weights to specific variables
    vWt00 = vWeightArray[0,0]
    vWt01 = vWeightArray[0,1]
    vWt10 = vWeightArray[1,0]       
    vWt11 = vWeightArray[1,1]        
    # Obtain the activations of the hidden nodes    
    hiddenActivation0 = computeSingleNeuronActivation(wWt00, wWt01, input0, input1)
    hiddenActivation1 = computeSingleNeuronActivation(wWt10, wWt11, input0, input1)

    # Obtain the activations of the output nodes    
    outputActivation0 = computeSingleNeuronActivation(vWt00, vWt01, hiddenActivation0, hiddenActivation1)
    outputActivation1 = computeSingleNeuronActivation(vWt10, vWt11, hiddenActivation0, hiddenActivation1)
               
    actualAllNodesOutputList = (hiddenActivation0, hiddenActivation1, outputActivation0, outputActivation1)                                                                                                
    return (actualAllNodesOutputList)
    
def PrintAndTraceBackpropagateOutputToHidden (alpha, nu, errorList, 
actualAllNodesOutputList, transFuncDerivList, deltaVWtArray, vWeightArray, newHiddenWeightArray):

    hiddenNode0 = actualAllNodesOutputList[0]
    hiddenNode1 = actualAllNodesOutputList[1]
    outputNode0 = actualAllNodesOutputList[2]    
    outputNode1 = actualAllNodesOutputList[3]
    transFuncDeriv0 = transFuncDerivList[0]
    transFuncDeriv1 = transFuncDerivList[1]
    deltaVWt00 = deltaVWtArray[0,0]
    deltaVWt01 = deltaVWtArray[0,1]
    deltaVWt10 = deltaVWtArray[1,0]
    deltaVWt11 = deltaVWtArray[1,1]
    error0 = errorList[0]
    error1 = errorList[1]                 
        
    print ' '
    print 'In Print and Trace for Backpropagation: Hidden to Output Weights'
    print '  Assuming alpha = 1'
    print ' '
    print '  The hidden node activations are:'
    print '    Hidden node 0: ', '  %.4f' % hiddenNode0, '  Hidden node 1: ', '  %.4f' % hiddenNode1   
    print ' '
    print '  The output node activations are:'
    print '    Output node 0: ', '  %.3f' % outputNode0, '   Output node 1: ', '  %.3f' % outputNode1       
    print ' ' 
    print '  The transfer function derivatives are: '
    print '    Deriv-F(0): ', '     %.3f' % transFuncDeriv0, '   Deriv-F(1): ', '     %.3f' % transFuncDeriv1

    print ' ' 
    print 'The computed values for the deltas are: '
    print '                nu  *  error  *   trFncDeriv *   hidden'
    print '  deltaVWt00 = ',' %.2f' % nu, '* %.4f' % error0, ' * %.4f' % transFuncDeriv0, '  * %.4f' % hiddenNode0
    print '  deltaVWt01 = ',' %.2f' % nu, '* %.4f' % error1, ' * %.4f' % transFuncDeriv1, '  * %.4f' % hiddenNode0                       
    print '  deltaVWt10 = ',' %.2f' % nu, '* %.4f' % error0, ' * %.4f' % transFuncDeriv0, '  * %.4f' % hiddenNode1
    print '  deltaVWt11 = ',' %.2f' % nu, '* %.4f' % error1, ' * %.4f' % transFuncDeriv1, '  * %.4f' % hiddenNode1
    print ' '
    print 'Values for the hidden-to-output connection weights:'
    print '           Old:     New:      nu*Delta:'
    print '[0,0]:   %.4f' % vWeightArray[0,0], '  %.4f' % newHiddenWeightArray[0,0], '  %.4f' % deltaVWtArray[0,0]
    print '[0,1]:   %.4f' % vWeightArray[0,1], '  %.4f' % newHiddenWeightArray[0,1], '  %.4f' % deltaVWtArray[0,1]
    print '[1,0]:   %.4f' % vWeightArray[1,0], '  %.4f' % newHiddenWeightArray[1,0], '  %.4f' % deltaVWtArray[1,0]
    print '[1,1]:   %.4f' % vWeightArray[1,1], '  %.4f' % newHiddenWeightArray[1,1], '  %.4f' % deltaVWtArray[1,1]
    
    
def BackpropagateOutputToHidden (alpha, eta, errorList, actualAllNodesOutputList, vWeightArray):
    error0 = errorList[0]
    error1 = errorList[1]
    
    vWt00 = vWeightArray[0,0]
    vWt01 = vWeightArray[0,1]
    vWt10 = vWeightArray[1,0]       
    vWt11 = vWeightArray[1,1]  
    
    hiddenNode0 = actualAllNodesOutputList[0]
    hiddenNode1 = actualAllNodesOutputList[1]
    outputNode0 = actualAllNodesOutputList[2]    
    outputNode1 = actualAllNodesOutputList[3]
    
        
    transFuncDeriv0 = computeTransferFnctnDeriv(outputNode0, alpha) 
    transFuncDeriv1 = computeTransferFnctnDeriv(outputNode1, alpha)
    transFuncDerivList = (transFuncDeriv0, transFuncDeriv1) 

    # Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
    #   and so is not included explicitly in these equations 
    
    # The equation for the actual dependence of the Summed Squared Error on a given hidden-to-output weight v(h,o) is:
    #   partial(SSE)/partial(v(h,o)) = -alpha*E(o)*F(o)*[1-F(o)]*H(h)
    # The transfer function derivative (transFuncDeriv) returned from computeTransferFnctnDeriv is given as:
    #   transFuncDeriv =  alpha*NeuronOutput*(1.0 -NeuronOutput)
    # Therefore, we can write the equation for the partial(SSE)/partial(v(h,o)) as
    #   partial(SSE)/partial(v(h,o)) = E(o)*transFuncDeriv*H(h)
    #   The parameter alpha is included in transFuncDeriv

    partialSSE_w_Vwt00 = -error0*transFuncDeriv0*hiddenNode0                                                             
    partialSSE_w_Vwt01 = -error1*transFuncDeriv1*hiddenNode0
    partialSSE_w_Vwt10 = -error0*transFuncDeriv0*hiddenNode1
    partialSSE_w_Vwt11 = -error0*transFuncDeriv0*hiddenNode0                                                                                                                                
                                                                                                                                                                                                                                                
    deltaVWt00 = -eta*partialSSE_w_Vwt00
    deltaVWt01 = -eta*partialSSE_w_Vwt01        
    deltaVWt10 = -eta*partialSSE_w_Vwt10
    deltaVWt11 = -eta*partialSSE_w_Vwt11 
    deltaVWtArray = np.array([[deltaVWt00, deltaVWt01],[deltaVWt10, deltaVWt11]])
    

    vWt00 = vWt00+deltaVWt00
    vWt01 = vWt01+deltaVWt01
    vWt10 = vWt10+deltaVWt10
    vWt11 = vWt11+deltaVWt11 
    
    newHiddenWeightArray = np.array([[vWt00, vWt01], [vWt10, vWt11]])                                                                      
    return (newHiddenWeightArray);     
    
def BackpropagateHiddenToInput (alpha, eta, errorList, actualAllNodesOutputList, inputDataList, vWeightArray, wWeightArray):

# The first step here applies a backpropagation-based weight change to the input-to-hidden wts w. 
# Core equation for the second part of backpropagation: 
# d(SSE)/dw(i,h) = -eta*alpha*F(h)(1-F(h))*Input(i)*sum(v(h,o)*Error(o))
# where:
# -- SSE = sum of squared errors, and only the error associated with a given output node counts
# -- w(i,h) is the connection weight w between the input node i and the hidden node h
# -- v(h,o) is the connection weight v between the hidden node h and the output node o
# -- alpha is the scaling term within the transfer function, often set to 1 
# ---- (this is included in transfFuncDeriv) 
# -- Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
# -- F = transfer function, here using the sigmoid transfer function
# ---- NOTE: in this second step, the transfer function is applied to the output of the hidden node,
# ------ so that F = F(h)
# -- Hidden(h) = the output of hidden node h (used in computing the derivative of the transfer function). 
# -- Input(i) = the input at node i.

# Note that the training rate parameter is assigned in main; Greek letter "eta," looks like n, 
#   scales amount of change to connection weight

# Unpack the errorList and the vWeightArray

# We will DECREMENT the connection weight v by a small amount proportional to the derivative eqn
#   of the SSE w/r/t the weight w. 
# This means, since there is a minus sign in that derivative, that we will add a small amount. 
# (Decrementing is -, applied to a (-), which yields a positive.)

# For the actual derivation of this equation with MATCHING VARIABLE NAMES (easy to understand), 
#   please consult: Brain-Based Computing, by AJ Maren (under development, Jan., 2017). Chpt. X. 
#   (Meaning: exact chapter is still TBD.) 
# For the latest updates, etc., please visit: www.aliannajmaren.com

# Note that the training rate parameter is assigned in main; Greek letter "eta," looks like n, 
#   scales amount of change to connection weight

# Unpack the errorList and the vWeightArray
    error0 = errorList[0]
    error1 = errorList[1]
    
    vWt00 = vWeightArray[0,0]
    vWt01 = vWeightArray[0,1]
    vWt10 = vWeightArray[1,0]       
    vWt11 = vWeightArray[1,1]  
    
    wWt00 = wWeightArray[0,0]
    wWt01 = wWeightArray[0,1]
    wWt10 = wWeightArray[1,0]       
    wWt11 = wWeightArray[1,1] 
    
    inputNode0 = inputDataList[0] 
    inputNode1 = inputDataList[1]         
    hiddenNode0 = actualAllNodesOutputList[0]
    hiddenNode1 = actualAllNodesOutputList[1]
    outputNode0 = actualAllNodesOutputList[2]    
    outputNode1 = actualAllNodesOutputList[3]
    
# For the second step in backpropagation (computing deltas on the input-to-hidden weights)
#   the transfer function derivative is applied to the output at the hidden node, not
#   to the output at the output node        
    transFuncDeriv0 = computeTransferFnctnDeriv(hiddenNode0, alpha) 
    transFuncDeriv1 = computeTransferFnctnDeriv(hiddenNode1, alpha)
    transFuncDerivList = (transFuncDeriv0, transFuncDeriv1) 
               
# Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
#   and so is not included explicitly in these equations
    partialSSE_w_Wwt00 = -transFuncDeriv0*inputNode0*(vWt00*error0 + vWt01*error1)                                                             
    partialSSE_w_Wwt01 = -transFuncDeriv1*inputNode0*(vWt10*error0 + vWt11*error1)
    partialSSE_w_Wwt10 = -transFuncDeriv0*inputNode1*(vWt00*error0 + vWt01*error1)
    partialSSE_w_Wwt11 = -transFuncDeriv1*inputNode1*(vWt10*error0 + vWt11*error1)                                                                                                    
                                                                                                                                                                                                                                                
    deltaWWt00 = -eta*partialSSE_w_Wwt00
    deltaWWt01 = -eta*partialSSE_w_Wwt01        
    deltaWWt10 = -eta*partialSSE_w_Wwt10
    deltaWWt11 = -eta*partialSSE_w_Wwt11 
    deltaWWtArray = np.array([[deltaWWt00, deltaWWt01],[deltaWWt10, deltaWWt11]])

    wWt00 = wWt00+deltaWWt00
    wWt01 = wWt01+deltaWWt01
    wWt10 = wWt10+deltaWWt10
    wWt11 = wWt11+deltaWWt11 
    
    newWWeightArray = np.array([[wWt00, wWt01], [wWt10, wWt11]])                                                                    
    return (newWWeightArray)
    
def main():

####################################################################################################
# Obtain unit array size in terms of array_length (M) and layers (N)
####################################################################################################                
    

    # Parameter definitions, to be replaced with user inputs
    alpha = 1.0             # parameter governing steepness of sigmoid transfer function
    summedInput = 1
    maxNumIterations = 1000 # temporarily set to 10 for testing
    eta = 0.5               # training rate     
    arraySizeList = list() # empty list
       
    arraySizeList = obtainNeuralNetworkSizeSpecs ()
    # Unpack the list; ascribe the various elements of the list to the sizes of different network layers    
    #unnecessary
    inputArrayLength = arraySizeList[0]
    hiddenArrayLength = arraySizeList [1]
    outputArrayLength = arraySizeList [2]                        

    # Initialize the training list
    trainingDataList = (0,0,0,0,0)
           

    wWeightArraySizeList = (inputArrayLength, hiddenArrayLength)
    vWeightArraySizeList = (hiddenArrayLength, outputArrayLength)    

    wWeightArray = initializeWeightArray (wWeightArraySizeList)
    vWeightArray = initializeWeightArray (vWeightArraySizeList)

    #this is not useful
    initialWWeightArray = wWeightArray[:]
    initialVWeightArray = vWeightArray[:]    

    print
    print 'The initial weights for this neural network are:'
    print '     Input-to-Hidden                       Hidden-to-Output'
    print 'w(0,0) = %.3f   w(0,1) = %.3f         v(0,0) = %.3f   v(0,1) = %.3f' % (initialWWeightArray[0,0], 
    initialWWeightArray[0,1], initialVWeightArray[0,0], initialVWeightArray[0,1])
    print 'w(1,0) = %.3f   w(1,1) = %.3f         v(1,0) = %.3f   v(1,1) = %.3f' % (initialWWeightArray[1,0], 
    initialWWeightArray[1,1], initialVWeightArray[1,0], initialVWeightArray[1,1])        
#0000000000000000000000000000000
    #move up in the program to the other constants
    epsilon = 0.2
    iteration = 0
    SSE_InitialTotal = 0.0
        
    # Initialize an array of SSE values
    SSE_Array = [100.0,100.0,100.0,100.0]
    SSE_InitialArray = [0,0,0,0]
    
    # Before starting the training run, compute the initial SSE Total 
    #   (sum across SSEs for each training data set)     
    ###SINGLE FEED FORWARD PASS
    # Compute a single feed-forward pass and obtain the Actual Outputs for zeroth data set
    inputDataList = (0, 0)           
    actualAllNodesOutputList = ComputeSingleFeedforwardPass (inputDataList, wWeightArray, vWeightArray)        
    actualOutput0 = actualAllNodesOutputList [2]
    actualOutput1 = actualAllNodesOutputList [3] 
    error0 = 1.0 - actualOutput0
    error1 = 0.0 - actualOutput1
    SSE_InitialArray[0] = error0**2 + error1**2

    # Compute a single feed-forward pass and obtain the Actual Outputs for first data set
    inputDataList = (0, 0)           
    actualAllNodesOutputList = ComputeSingleFeedforwardPass (inputDataList, wWeightArray, vWeightArray)        
    actualOutput0 = actualAllNodesOutputList [2]
    actualOutput1 = actualAllNodesOutputList [3] 
    error0 = 0.0 - actualOutput0
    error1 = 1.0 - actualOutput1
    SSE_InitialArray[1] = error0**2 + error1**2
                                                        
    # Compute a single feed-forward pass and obtain the Actual Outputs for second data set
    inputDataList = (0, 0)           
    actualAllNodesOutputList = ComputeSingleFeedforwardPass (inputDataList, wWeightArray, vWeightArray)        
    actualOutput0 = actualAllNodesOutputList [2]
    actualOutput1 = actualAllNodesOutputList [3] 
    error0 = 0.0 - actualOutput0
    error1 = 1.0 - actualOutput1
    SSE_InitialArray[2] = error0**2 + error1**2
        
    # Compute a single feed-forward pass and obtain the Actual Outputs for third data set
    inputDataList = (1, 1)           
    actualAllNodesOutputList = ComputeSingleFeedforwardPass (inputDataList, wWeightArray, vWeightArray)        
    actualOutput0 = actualAllNodesOutputList [2]
    actualOutput1 = actualAllNodesOutputList [3] 
    error0 = 1.0 - actualOutput0
    error1 = 0.0 - actualOutput1
    SSE_InitialArray[3] = error0**2 + error1**2

    # Initialize an array of SSE values
    SSE_InitialTotal = SSE_InitialArray[0] + SSE_InitialArray[1] +SSE_InitialArray[2] + SSE_InitialArray[3]                                                 
                                                                                                                                                
    while iteration < maxNumIterations: 
           

    ####################################################################################################
    # Next step - Obtain a single set of input values for the X-OR problem; two integers - can be 0 or 1
    ####################################################################################################                

        # Randomly select one of four training sets; the inputs will be randomly assigned to 0 or 1
        trainingDataList = obtainRandomXORTrainingValues () 
        input0 = trainingDataList[0]
        input1 = trainingDataList[1] 
        desiredOutput0 = trainingDataList[2]
        desiredOutput1 = trainingDataList[3]
        setNumber = trainingDataList[4]       
        print ' '
        print 'Randomly selecting XOR inputs for XOR, identifying desired outputs for this training pass:'
        print '          Input0 = ', input0,         '            Input1 = ', input1   
        print ' Desired Output0 = ', desiredOutput0, '   Desired Output1 = ', desiredOutput1    
        print ' '
         

    ####################################################################################################
    # Compute a single feed-forward pass
    ####################################################################################################                
 
        # Initialize the error list
        errorList = (0,0)
    
        # Initialize the actualOutput list
        actualAllNodesOutputList = (0,0,0,0)     

        # Create the inputData list      
        inputDataList = (input0, input1)         
    
        # Compute a single feed-forward pass and obtain the Actual Outputs
        actualAllNodesOutputList = ComputeSingleFeedforwardPass (inputDataList, wWeightArray, vWeightArray)

        # Assign the hidden and output values to specific different variables
        actualHiddenOutput0 = actualAllNodesOutputList [0] 
        actualHiddenOutput1 = actualAllNodesOutputList [1] 
        actualOutput0 = actualAllNodesOutputList [2]
        actualOutput1 = actualAllNodesOutputList [3] 
    
        # Determine the error between actual and desired outputs

        error0 = desiredOutput0 - actualOutput0
        error1 = desiredOutput1 - actualOutput1
        errorList = (error0, error1)
    
        # Compute the Summed Squared Error, or SSE
        SSEInitial = error0**2 + error1**2

        # Print the Summed Squared Error  
        print 'Initial SSE = %.6f' % SSEInitial    
  
        # Perform first part of the backpropagation of weight changes    
        newVWeightArray = BackpropagateOutputToHidden (alpha, eta, errorList, actualAllNodesOutputList, vWeightArray)
        newWWeightArray = BackpropagateHiddenToInput (alpha, eta, errorList, actualAllNodesOutputList, inputDataList, vWeightArray, wWeightArray)
    
        # Assign the old hidden-to-output weight array to be the same as what was returned from the BP weight update
        vWeightArray = newVWeightArray[:]
    
        # Assign the old input-to-hidden weight array to be the same as what was returned from the BP weight update
        wWeightArray = newWWeightArray[:]
    
        # Run the computeSingleFeedforwardPass again, to compare the results after just adjusting the hidden-to-output weights
        newAllNodesOutputList = ComputeSingleFeedforwardPass (inputDataList, wWeightArray, vWeightArray)         
        newOutput0 = newAllNodesOutputList [2]
        newOutput1 = newAllNodesOutputList [3] 

        # Determine the new error between actual and desired outputs
        newError0 = desiredOutput0 - newOutput0
        newError1 = desiredOutput1 - newOutput1
        newErrorList = (newError0, newError1)

        # Compute the new Summed Squared Error, or SSE
        SSE = newError0**2 + newError1**2

        # Print the Summed Squared Error  
        print 'New SSE = %.6f' % SSE


        # Assign the SSE to the SSE for the appropriate training set
        SSE_Array[setNumber] = SSE
        
        # Assign the new errors to the error list             
        errorList = newErrorList[:]

        # Compute the new sum of SSEs (across all the different training sets)
        #   ... this will be different because we've changed one of the SSE's
        SSE_Total = SSE_Array[0] + SSE_Array[1] +SSE_Array[2] + SSE_Array[3] 

        print ' '
        print 'Iteration number ', iteration
        iteration = iteration + 1

        if SSE_Total < epsilon:
            break
    print 'Out of while loop'     


    print ' '
    print 'The initial weights for this neural network are:'
    print '     Input-to-Hidden                       Hidden-to-Output'
    print 'w(0,0) = %.3f   w(0,1) = %.3f         v(0,0) = %.3f   v(0,1) = %.3f' % (initialWWeightArray[0,0], 
    initialWWeightArray[0,1], initialVWeightArray[0,0], initialVWeightArray[0,1])
    print 'w(1,0) = %.3f   w(1,1) = %.3f         v(1,0) = %.3f   v(1,1) = %.3f' % (initialWWeightArray[1,0], 
    initialWWeightArray[1,1], initialVWeightArray[1,0], initialVWeightArray[1,1])        

                                                                                    
    print ' '
    print 'The final weights for this neural network are:'
    print '     Input-to-Hidden                       Hidden-to-Output'
    print 'w(0,0) = %.3f   w(0,1) = %.3f         v(0,0) = %.3f   v(0,1) = %.3f' % (wWeightArray[0,0], 
    wWeightArray[0,1], vWeightArray[0,0], vWeightArray[0,1])
    print 'w(1,0) = %.3f   w(1,1) = %.3f         v(1,0) = %.3f   v(1,1) = %.3f' % (wWeightArray[1,0], 
    wWeightArray[1,1], vWeightArray[1,0], vWeightArray[1,1])        
                                                                                    
   
    # Print the SSE's at the beginning of training
    print ' '
    print 'The SSE values at the beginning of training were: '
    print '  SSE_Initial[0] = %.4f' % SSE_InitialArray[0]
    print '  SSE_Initial[1] = %.4f' % SSE_InitialArray[1]
    print '  SSE_Initial[2] = %.4f' % SSE_InitialArray[2]
    print '  SSE_Initial[3] = %.4f' % SSE_InitialArray[3]    
    print ' '
    print 'The total of the SSE values at the beginning of training is %.4f' % SSE_InitialTotal 


    # Print the SSE's at the end of training
    print ' '
    print 'The SSE values at the end of training were: '
    print '  SSE[0] = %.4f' % SSE_Array[0]
    print '  SSE[1] = %.4f' % SSE_Array[1]
    print '  SSE[2] = %.4f' % SSE_Array[2]
    print '  SSE[3] = %.4f' % SSE_Array[3]    
    print ' '
    print 'The total of the SSE values at the end of training is %.4f' % SSE_Total                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
# Print comparison of previous and new outputs             
#    print ' ' 
#    print 'Values for the new outputs compared with previous, given only a partial backpropagation training:'
#    print '     Old:', '   ', 'New:', '   ', 'nu*Delta:'
#    print 'Output 0:  Desired = ', desiredOutput0, 'Old actual =  %.4f' % actualOutput0, 'Newactual  %.4f' % newOutput0
#    print 'Output 1:  Desired = ', desiredOutput1, 'Old actual =  %.4f' % actualOutput1, 'Newactual  %.4f' % newOutput1                                                                         
                                                            
                     
####################################################################################################
# Conclude specification of the MAIN procedure
####################################################################################################                
    
if __name__ == "__main__": main()