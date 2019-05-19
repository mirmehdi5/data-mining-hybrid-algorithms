package com.ec.lab;

/**
 * This project compares various classifier algorithms
 * Please read the readme file to know how to run the code
 * @author MIR
 *
 */
public class App 
{    
    public static void main( String[] args ) throws Exception {
    	
    	String inputFileSet = new String();
    	
    		if (args[0].equals("-i")) 
    			inputFileSet = "data/"+args[1];
			
        AlgorithmswithAdaBoost objAlgorithm = new AlgorithmswithAdaBoost();
        objAlgorithm.callAlgo(inputFileSet);
        
        AlgorithmswithoutAdaBoost objwithoutAlgorithm = new AlgorithmswithoutAdaBoost();
        objwithoutAlgorithm.callAlgo(inputFileSet);
    }
}
