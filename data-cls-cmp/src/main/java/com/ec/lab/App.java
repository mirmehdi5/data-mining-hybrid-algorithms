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
		System.out.println("Algorithms with Adaboost starts-----------------------");
        AlgorithmswithAdaBoost objAlgorithm = new AlgorithmswithAdaBoost();
        objAlgorithm.callAlgo(inputFileSet);
        System.out.println("Algorithms with Adaboost ends-----------------------");
        
        System.out.println("Algorithms without Adaboost starts-----------------------");
        AlgorithmswithoutAdaBoost objwithoutAlgorithm = new AlgorithmswithoutAdaBoost();
        objwithoutAlgorithm.callAlgo(inputFileSet);
        System.out.println("Algorithms without Adaboost ends-----------------------");
    }
}
