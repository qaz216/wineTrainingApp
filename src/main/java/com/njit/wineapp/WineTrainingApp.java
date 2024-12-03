package com.njit.wineapp;

import java.io.IOException;
import java.util.Properties;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class WineTrainingApp {
	private static final String MASTER_NODE = "local[*]";
	private Properties prop = null;

	
	public WineTrainingApp() {
		this.prop = new Properties();
		try {
			this.prop.load(WineTrainingApp.class.getClassLoader().getResourceAsStream("application.properties"));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	
    public static void main(String[] args) throws IOException {
    	WineTrainingApp app = new WineTrainingApp();
    	String mode = app.getMode();
		System.out.println("Mode: " + mode);
		if(mode.equals("train_model")) {
			//app.trainModel();
			app.trainModel2();
			System.exit(0);
		}
		else if(mode.equals("run_model")) {
			//app.runModel();
			app.runModel2();
			System.exit(0);
		}
        SparkConf conf = new SparkConf().setMaster(MASTER_NODE).setAppName("WineTrainingApp");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().appName("WineTrainingApp").getOrCreate();

        // Load the training dataset from S3
        Dataset<Row> trainingData = spark.read().format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("TrainingDataset.csv");
                //.load("s3://<your-s3-bucket>/TrainingDataset.csv");

        // Prepare the feature columns
        String[] featureColumns = {"fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                "pH", "sulphates", "alcohol"};
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureColumns)
                .setOutputCol("features");

        // Split the data into training and validation sets
        //Dataset<Row>[] splits = trainingData.randomSplit(new double[]{0.8, 0.2}, 42L);
        //Dataset<Row> trainData = splits[0];
        //Dataset<Row> validationData = splits[1];
        
        Dataset<Row> testingDf1 = getDataFrame(spark, true, "TrainingDataset.csv").cache();

        // Create a LogisticRegression model
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8)
                .setFamily("multinomial")
                .setLabelCol("label");
                //.setLabelCol("quality");

        // Create a pipeline with feature transformation and model
        //Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{assembler, lr});
        LogisticRegression logReg = lr;
        //LogisticRegression logReg = new LogisticRegression().setMaxIter(100).setRegParam(0.0); 
        Pipeline pl1 = new Pipeline();
        pl1.setStages(new PipelineStage[]{logReg});

        Dataset<Row> lblFeatureDf = getDataFrame(spark, true, "TrainingDataset.csv").cache();
        // Train the model
        PipelineModel model = pl1.fit(lblFeatureDf);
        //PipelineModel model = pipeline.fit(lblFeatureDf);
        //PipelineModel model = pipeline.fit(trainData);

        // Evaluate the model on the validation dataset
        Dataset<Row> predictions = model.transform(testingDf1);
        //Dataset<Row> predictions = model.transform(validationData);
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                //.setLabelCol("quality")
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double f1Score = evaluator.evaluate(predictions);
        System.out.println("F1 score on validation data: " + f1Score);

        // Save the trained model
        model.write().overwrite().save("wineQualityModel");

        jsc.close();
    }
    
    private void runModel2() {
    	String appName = getAppName();
        SparkConf conf = new SparkConf().setMaster(MASTER_NODE).setAppName(appName);
        JavaSparkContext jsc = new JavaSparkContext(conf);
        SparkSession spark =  SparkSession.builder().appName(appName).getOrCreate();
        System.out.println("TestingDataSet Metrics \n");
        PipelineModel pipelineModel = PipelineModel.load(appName);
        Dataset<Row> testDf = getDataFrame(spark, true, "ValidationDataset.csv").cache();
        Dataset<Row> predictionDF = pipelineModel.transform(testDf).cache();
        predictionDF.select("features", "label", "prediction").show(5, false);
        printMertics(predictionDF);	
	}


	private void trainModel2() {
		String appName = getAppName();
        SparkConf conf = new SparkConf().setMaster(MASTER_NODE).setAppName(appName);
        JavaSparkContext jsc = new JavaSparkContext(conf);
        SparkSession spark =  SparkSession.builder().appName(appName).getOrCreate();
        
        Dataset<Row> lblFeatureDf = getDataFrame(spark, true, "TrainingDataset.csv").cache();
        LogisticRegression logReg = new LogisticRegression().setMaxIter(100).setRegParam(0.0);

        Pipeline pl1 = new Pipeline();
        pl1.setStages(new PipelineStage[]{logReg});

        PipelineModel model1 = pl1.fit(lblFeatureDf);
        
        LogisticRegressionModel lrModel = (LogisticRegressionModel) (model1.stages()[0]);
        // System.out.println("Learned LogisticRegressionModel:\n" + lrModel.summary().accuracy());
        LogisticRegressionTrainingSummary trainingSummary = lrModel.summary();
        double accuracy = trainingSummary.accuracy();
        double falsePositiveRate = trainingSummary.weightedFalsePositiveRate();
        double truePositiveRate = trainingSummary.weightedTruePositiveRate();
        double fMeasure = trainingSummary.weightedFMeasure();
        double precision = trainingSummary.weightedPrecision();
        double recall = trainingSummary.weightedRecall();

        System.out.println();
        System.out.println("Training DataSet Metrics ");

        System.out.println("Accuracy: " + accuracy);
        System.out.println("FPR: " + falsePositiveRate);
        System.out.println("TPR: " + truePositiveRate);
        System.out.println("F-measure: " + fMeasure);
        System.out.println("Precision: " + precision);
        System.out.println("Recall: " + recall);


        Dataset<Row> testingDf1 = getDataFrame(spark, true, "ValidationDataset.csv").cache();

        Dataset<Row> results = model1.transform(testingDf1);
        

        System.out.println("\n Validation Training Set Metrics");
        results.select("features", "label", "prediction").show(5, false);
        printMertics(results);

        try {
            model1.write().overwrite().save(appName);
        } catch (IOException e) {
        	e.printStackTrace();
        }

		
	}
    
    public void printMertics(Dataset<Row> predictions) {
        System.out.println();
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();
        evaluator.setMetricName("accuracy");
        System.out.println("The accuracy of the model is " + evaluator.evaluate(predictions));

        evaluator.setMetricName("accuracy");
        double accuracy1 = evaluator.evaluate(predictions);
        System.out.println("Test Error = " + (1.0 - accuracy1));

        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(predictions);

        evaluator.setMetricName("weightedPrecision");
        double weightedPrecision = evaluator.evaluate(predictions);

        evaluator.setMetricName("weightedRecall");
        double weightedRecall = evaluator.evaluate(predictions);

        System.out.println("Accuracy: " + accuracy1);
        System.out.println("F1: " + f1);
        System.out.println("Precision: " + weightedPrecision);
        System.out.println("Recall: " + weightedRecall);

    }


	private void runModel() {
        SparkConf conf = new SparkConf().setMaster(MASTER_NODE).setAppName("WineTrainingApp");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        SparkSession spark =  SparkSession.builder().appName("WineTrainingApp").getOrCreate();
		
        PipelineModel model = PipelineModel.load("wineQualityModel");
		Dataset<Row> trainingFrame = getDataFrame(spark, true, "TrainingDataset.csv").cache();
		
		Dataset<Row> predictions = model.transform(trainingFrame);
		
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                //.setLabelCol("quality")
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double f1Score = evaluator.evaluate(predictions);
        System.out.println("F1 score on test data: " + f1Score);

        jsc.close();
	}


	private void trainModel() throws IOException {
    	//SparkSession spark = getSparkSession();
        SparkConf conf = new SparkConf().setMaster(MASTER_NODE).setAppName("WineTrainingApp");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        SparkSession spark =  SparkSession.builder().appName("WineTrainingApp").getOrCreate();
		Dataset<Row> trainingFrame = getDataFrame(spark, true, "TrainingDataset.csv").cache();
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8)
                .setFamily("multinomial")
                .setLabelCol("label");
        LogisticRegression logReg = new LogisticRegression().setMaxIter(100).setRegParam(0.0); 
        Pipeline pl1 = new Pipeline();
        pl1.setStages(new PipelineStage[]{lr});
        PipelineModel model = pl1.fit(trainingFrame);
        
        Dataset<Row> validationFram = getDataFrame(spark, true, "ValidationDataset.csv").cache();
        
        Dataset<Row> results = model.transform(validationFram);
        
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1Score = evaluator.evaluate(results);
        System.out.println("F1 score on validation data: " + f1Score);

        // Save the trained model
        //model.write().overwrite().save("s3://<your-s3-bucket>/wineQualityModel");
        model.write().overwrite().save("wineQualityModel");

        jsc.close();

		System.exit(0);
		
	}


	private SparkSession getSparkSession() {
        SparkConf conf = new SparkConf().setMaster(MASTER_NODE).setAppName("WineTrainingApp");
        //JavaSparkContext jsc = new JavaSparkContext(conf);
        return SparkSession.builder().appName("WineTrainingApp").getOrCreate();
	}


	public static Dataset<Row> getDataFrame(SparkSession spark, boolean transform, String name) {

        Dataset<Row> validationDf = spark.read().format("csv").option("header", "true")
                .option("multiline", true).option("sep", ";").option("quote", "\"")
                .option("dateFormat", "M/d/y").option("inferSchema", true).load(name);
        
        validationDf.printSchema();


        validationDf = validationDf.withColumnRenamed("fixed acidity", "fixed_acidity")
                .withColumnRenamed("volatile acidity", "volatile_acidity")
                .withColumnRenamed("citric acid", "citric_acid")
                .withColumnRenamed("residual sugar", "residual_sugar")
                .withColumnRenamed("chlorides", "chlorides")
                .withColumnRenamed("free sulfur dioxide", "free_sulfur_dioxide")
                .withColumnRenamed("total sulfur dioxide", "total_sulfur_dioxide")
                .withColumnRenamed("density", "density").withColumnRenamed("pH", "pH")
                .withColumnRenamed("sulphates", "sulphates").withColumnRenamed("alcohol", "alcohol")
                .withColumnRenamed("quality", "label");

        validationDf.show(5);


        Dataset<Row> lblFeatureDf = validationDf.select("label", "alcohol", "sulphates", "pH",
                "density", "free_sulfur_dioxide", "total_sulfur_dioxide", "chlorides", "residual_sugar",
                "citric_acid", "volatile_acidity", "fixed_acidity");

        lblFeatureDf = lblFeatureDf.na().drop().cache();

        VectorAssembler assembler =
                new VectorAssembler().setInputCols(new String[]{"alcohol", "sulphates", "pH", "density",
                        "free_sulfur_dioxide", "total_sulfur_dioxide", "chlorides", "residual_sugar",
                        "citric_acid", "volatile_acidity", "fixed_acidity"}).setOutputCol("features");

        if (transform)
            lblFeatureDf = assembler.transform(lblFeatureDf).select("label", "features");


        return lblFeatureDf;
    }
    
	private String getMode() {
		return this.prop.getProperty("app.mode");
	}

	private String getAppName() {
		return this.prop.getProperty("app.name");
	}

}