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
  private static final String ACCESS_KEY_ID = System.getProperty("ACCESS_KEY_ID");
  private static final String SECRET_KEY = System.getProperty("SECRET_KEY");
  
  private Properties prop = null;

  public WineTrainingApp() {
    this.prop = new Properties();
    try {
      this.prop.load(WineTrainingApp.class.getClassLoader().getResourceAsStream("application.properties"));
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) throws Exception {
    // Create Wine Training Application
    WineTrainingApp app = new WineTrainingApp();
    String mode = app.getMode();
    System.out.println("Mode: " + mode);

    if (mode.equals("train_model")) {
      // Train Model (Parallelize)
      app.trainModel();
      System.exit(0);
    } else if (mode.equals("run_model")) {
      // Run Model
      app.runModel();
      System.exit(0);
    }
  }

  private void runModel() {
    String appName = getAppName();
    String validationData = getValidationSet();
    String bucket = getS3Bucket();
    validationData = bucket + validationData;
    SparkConf conf = new SparkConf().setMaster(MASTER_NODE).setAppName(appName);
    JavaSparkContext jsc = new JavaSparkContext(conf);
    SparkSession spark = SparkSession.builder().appName(appName).getOrCreate();
    if((ACCESS_KEY_ID != null && !ACCESS_KEY_ID.isEmpty()) && SECRET_KEY != null && !SECRET_KEY.isEmpty()) {
    	spark.sparkContext().hadoopConfiguration().set("fs.s3a.access.key", ACCESS_KEY_ID);
    	spark.sparkContext().hadoopConfiguration().set("fs.s3a.secret.key", SECRET_KEY);
    }
    PipelineModel pipelineModel = PipelineModel.load(appName);
    Dataset < Row > testDf = getFrame(spark, true, validationData).cache();
    Dataset < Row > predictionDF = pipelineModel.transform(testDf).cache();
    predictionDF.select("features", "label", "prediction").show(5, false);
    outputResults(predictionDF);
    jsc.close();
  }

  private void trainModel() throws Exception {
    // Configure Spark Session
    String appName = getAppName();
    SparkConf conf = new SparkConf().setMaster(MASTER_NODE).setAppName(appName)
      .set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem");
    JavaSparkContext jsc = new JavaSparkContext(conf);
    SparkSession spark = SparkSession.builder().appName(appName).master(MASTER_NODE).getOrCreate();
    if((ACCESS_KEY_ID != null && !ACCESS_KEY_ID.isEmpty()) && SECRET_KEY != null && !SECRET_KEY.isEmpty()) {
    	spark.sparkContext().hadoopConfiguration().set("fs.s3a.access.key", ACCESS_KEY_ID);
    	spark.sparkContext().hadoopConfiguration().set("fs.s3a.secret.key", SECRET_KEY);
    }

    // Get training data and provision regression
    String bucket = getS3Bucket();
    String traingSet = getTrainingSet();
    traingSet = bucket + traingSet;
    Dataset < Row > trainingDf = getFrame(spark, true, traingSet).cache();
    LogisticRegression regression = new LogisticRegression().setMaxIter(100).setRegParam(0.0);

    // Train Model
    Pipeline pipeline = new Pipeline();
    pipeline.setStages(new PipelineStage[] {
      regression
    });
    PipelineModel trainingModel = pipeline.fit(trainingDf);
    LogisticRegressionModel logRegModel = (LogisticRegressionModel)(trainingModel.stages()[0]);
    LogisticRegressionTrainingSummary ts = logRegModel.summary();

    // Output Summary Metrics
    System.out.println();
    System.out.println("Metrics: ");
    System.out.println("fpRate: " + ts.weightedFalsePositiveRate());
    System.out.println("tpRate: " + ts.weightedTruePositiveRate());
    System.out.println("accuracy: " + ts.accuracy());
    System.out.println("fMeasure: " + ts.weightedFMeasure());
    System.out.println("Precision: " + ts.weightedPrecision());
    System.out.println("Recall: " + ts.weightedRecall());


    String validationSet = getValidationSet();
    validationSet = bucket + validationSet;
    System.out.println("validation dataset: "+validationSet);
    Dataset < Row > validationFrame = getFrame(spark, true, validationSet).cache();

    Dataset < Row > results = trainingModel.transform(validationFrame);

    results.select("features", "label", "prediction").show(5, false);
    outputResults(results);
    
    String model = bucket + appName;

    trainingModel.write().overwrite().save(model);

    jsc.close();

  }

  public void outputResults(Dataset < Row > predictions) {
    MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();

    evaluator.setMetricName("weightedPrecision");
    double weightedPrecision = evaluator.evaluate(predictions);

    evaluator.setMetricName("weightedRecall");
    double weightedRecall = evaluator.evaluate(predictions);

    evaluator.setMetricName("accuracy");
    double accuracy = evaluator.evaluate(predictions);

    evaluator.setMetricName("f1");
    double f1 = evaluator.evaluate(predictions);

    System.out.println("weightedPrecision: " + weightedPrecision + " - weightedRecall: " +
      weightedRecall + " - accuracy: " + accuracy + " - f1: " + f1);

  }

  public static Dataset < Row > getFrame(SparkSession spark, boolean transform, String name) {
    // Initialize Frame Schema
    Dataset < Row > vdf = spark.read().format("csv").option("header", "true")
      .option("multiline", true).option("sep", ";").option("quote", "\"")
      .option("dateFormat", "M/d/y").option("inferSchema", true).load(name);

    // Print Schema
    vdf.printSchema();

    // Rename Fields
    vdf = vdf.withColumnRenamed("fixed acidity", "fixed_acidity")
      .withColumnRenamed("volatile acidity", "volatile_acidity")
      .withColumnRenamed("residual sugar", "residual_sugar")
      .withColumnRenamed("total sulfur dioxide", "total_sulfur_dioxide")
      .withColumnRenamed("citric acid", "citric_acid")
      .withColumnRenamed("chlorides", "chlorides")
      .withColumnRenamed("density", "density").withColumnRenamed("pH", "pH")
      .withColumnRenamed("free sulfur dioxide", "free_sulfur_dioxide")
      .withColumnRenamed("sulphates", "sulphates").withColumnRenamed("alcohol", "alcohol")
      .withColumnRenamed("quality", "label");
    vdf.show(5);

    // Get Label Dataframe
    Dataset < Row > labelDf = vdf.select("label", "alcohol", "sulphates", "pH",
      "density", "free_sulfur_dioxide", "total_sulfur_dioxide", "chlorides", "residual_sugar",
      "citric_acid", "volatile_acidity", "fixed_acidity");

    labelDf = labelDf.na().drop().cache();

    // Define Assembler
    VectorAssembler assembler =
      new VectorAssembler().setInputCols(new String[] {
        "alcohol",
        "sulphates",
        "pH",
        "density",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "chlorides",
        "residual_sugar",
        "citric_acid",
        "volatile_acidity",
        "fixed_acidity"
      }).setOutputCol("features");

    // Transform
    if (transform) {
      labelDf = assembler.transform(labelDf).select("label", "features");
    }

    return labelDf;
  }

  private String getMode() {
    return this.prop.getProperty("app.mode");
  }

  private String getAppName() {
    return this.prop.getProperty("app.name");
  }

  private String getTrainingSet() {
    return this.prop.getProperty("app.dataset.training");
  }

  private String getValidationSet() {
    return this.prop.getProperty("app.dataset.validation");
  }

  private String getS3Bucket() {
    return this.prop.getProperty("app.s3.bucket");
  }
}