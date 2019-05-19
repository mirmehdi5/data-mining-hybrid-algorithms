package com.ec.lab;

import java.io.File;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import java.util.concurrent.TimeUnit;

/**
 * This project compares various classifier algorithms
 * @author MIR
 *
 */
public class AlgorithmswithAdaBoost {
    public void callAlgo(String inputFileSet) throws Exception {
    	
    	ArffLoader loader1 = new ArffLoader();
        loader1.setFile(new File(inputFileSet));
        Instances data1 = loader1.getDataSet();
        data1.setClassIndex(data1.numAttributes() - 1);
        int seed = 1234;
        int folds = 10;
        Random rand = new Random(seed);
        Instances newData = new Instances(data1);
        newData.randomize(rand);
        if (newData.classAttribute().isNominal()) {
            newData.stratify(folds);
        }
 
        System.out.println("\n==============================================SVM algorithm Starts==============================================\n");
        
        Classifier svm_Classifier = new SMO();
        AdaBoostM1 svm_adaboost;
        svm_adaboost = new AdaBoostM1();
        
        
        long svm_train_time_Start = System.currentTimeMillis();
        svm_adaboost.setClassifier(svm_Classifier);
        svm_adaboost.buildClassifier(data1);
        long svm_train_time_End = System.currentTimeMillis();
        
        long svm_test_time_Start = System.currentTimeMillis();
        Evaluation eval_svm = new Evaluation(newData);
        for (int i = 0; i < folds; i++) {
            Instances test2 = newData.testCV(folds, i);
            eval_svm.evaluateModel(svm_adaboost, test2);
        }
        long svm_test_time_End = System.currentTimeMillis();

        System.out.println("\n SVM Classifier Training Time is\t\t"+ (svm_train_time_End-svm_train_time_Start) +"  milliseconds");
        System.out.println("\n SVM Classifier Testing Time is\t\t\t"+ (svm_test_time_End-svm_test_time_Start) +"  milliseconds");
        System.out.println("\nCorrectly Classified Instances\t\t\t"+eval_svm.correct());
        System.out.println("\nIncorrectly Classified Instances\t\t"+eval_svm.incorrect());
        System.out.println("\nPercentage of instances correctly classified \t"+eval_svm.pctCorrect()+"%");
        System.out.println("\nPrecision \t\t\t\t\t"+eval_svm.precision(0));
        System.out.println("\nRecall \t\t\t\t\t\t"+eval_svm.recall(0));
        System.out.println("\nFMeasure \t\t\t\t\t"+eval_svm.fMeasure(0));
        System.out.println("\n==============================================SVM algorithm Ends==============================================\n"); 

        System.out.println("\n==============================================NB algorithm Starts==============================================");
        
        Classifier nb_Classifier = new NaiveBayes();
        AdaBoostM1 nb_adaboost;
        nb_adaboost = new AdaBoostM1();
        
        long nb_train_time_Start = System.currentTimeMillis();
        nb_Classifier.buildClassifier(data1);
        nb_adaboost.setClassifier(nb_Classifier);
        nb_adaboost.buildClassifier(data1);
        long nb_train_time_End = System.currentTimeMillis();
        
        long nb_test_time_Start = System.currentTimeMillis();
        Evaluation eval_nb = new Evaluation(newData);
        for (int i = 0; i < folds; i++) {
            Instances test2 = newData.testCV(folds, i);
            eval_nb.evaluateModel(nb_adaboost, test2);
        }
        long nb_test_time_End = System.currentTimeMillis();
        
        System.out.println("\n NB Classifier Training Time is\t\t\t"+ (nb_train_time_End-nb_train_time_Start) +"  milliseconds");
        System.out.println("\n NB Classifier Testing Time is\t\t\t"+ (nb_test_time_End-nb_test_time_Start) +"  milliseconds");
        System.out.println("\nCorrectly Classified Instances\t\t\t"+eval_nb.correct());
        System.out.println("\nIncorrectly Classified Instances\t\t"+eval_nb.incorrect());
        System.out.println("\nPercentage of instances correctly classified \t"+eval_nb.pctCorrect()+"%");
        System.out.println("\nPrecision \t\t\t\t\t"+eval_nb.precision(0));
        System.out.println("\nRecall \t\t\t\t\t\t"+eval_nb.recall(0));
        System.out.println("\nFMeasure \t\t\t\t\t"+eval_nb.fMeasure(0));
        System.out.println("\n==============================================NB algorithm Ends==============================================\n"); 
    }
}