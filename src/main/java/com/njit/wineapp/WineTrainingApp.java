package com.njit.wineapp;

import java.io.IOException;
import java.util.Properties;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.apache.hadoop.shaded.org.apache.commons.text.StringEscapeUtils;
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
import java.io.FileWriter;

public class WineTrainingApp {
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
    } else if (mode.equals("run_model")) {
      // Run Model
      app.runModel();
      System.exit(0);
    }
  }

  private void runModel() throws Exception {
    String appName = getAppName();
    String validationData = getValidationSet();
    String bucket = getS3Bucket();
    String masterNode = getMasterNode();
    String accessKey = getAccessKey();
    String secretKey = getSecretKey();
    String sessionToken = getSessionToken();
    validationData = bucket + validationData;
    SparkConf conf = new SparkConf().setMaster(masterNode).setAppName(appName)
      .set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
      .set("spark.testing.memory", "571859200")
      .set("fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider");
    JavaSparkContext jsc = new JavaSparkContext(conf);
    SparkSession spark = SparkSession.builder().appName(appName).getOrCreate();
    spark.sparkContext().hadoopConfiguration().set("fs.s3a.access.key", accessKey);
    spark.sparkContext().hadoopConfiguration().set("fs.s3a.secret.key", secretKey);
    spark.sparkContext().hadoopConfiguration().set("fs.s3a.session.token", sessionToken);
    String model = bucket + appName + "/";
    System.out.println("model: " + model);
    PipelineModel pipelineModel = PipelineModel.load(model);
    Dataset < Row > testDf = getFrame(spark, true, validationData).cache();
    Dataset < Row > predictionDF = pipelineModel.transform(testDf).cache();
    outputResults(predictionDF);
    jsc.close();
  }

  private void trainModel() throws Exception {
    // Configure Spark Session
    String appName = getAppName();
    String masterNode = getMasterNode();
    String accessKey = getAccessKey();
    String secretKey = getSecretKey();
    String sessionToken = getSessionToken();
    SparkConf conf = new SparkConf().setMaster(masterNode).setAppName(appName)
      .set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
      .set("fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider");
    JavaSparkContext jsc = new JavaSparkContext(conf);
    SparkSession spark = SparkSession.builder().appName(appName).getOrCreate();
    spark.sparkContext().hadoopConfiguration().set("fs.s3a.access.key", accessKey);
    spark.sparkContext().hadoopConfiguration().set("fs.s3a.secret.key", secretKey);
    spark.sparkContext().hadoopConfiguration().set("fs.s3a.session.token", sessionToken);

    // Get training data and provision regression
    String bucket = getS3Bucket();
    String traingSet = getTrainingSet();
    traingSet = bucket + traingSet;
    Dataset < Row > trainingDf = getFrame(spark, true, traingSet).cache();
    LogisticRegression regression = new LogisticRegression().setMaxIter(200).setRegParam(0.0);

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
    System.out.println("validation dataset: " + validationSet);
    Dataset < Row > validationFrame = getFrame(spark, true, validationSet).cache();
    Dataset < Row > results = trainingModel.transform(validationFrame);

    String model = bucket + appName;

    trainingModel.write().overwrite().save(model);

    jsc.close();

  }

  public void outputResults(Dataset < Row > predictions) throws Exception {
    MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();

    evaluator.setMetricName("weightedPrecision");
    double weightedPrecision = evaluator.evaluate(predictions);

    evaluator.setMetricName("weightedRecall");
    double weightedRecall = evaluator.evaluate(predictions);

    evaluator.setMetricName("accuracy");
    double accuracy = evaluator.evaluate(predictions);

    evaluator.setMetricName("f1");
    double f1 = evaluator.evaluate(predictions);

    String s = "weightedPrecision: " + weightedPrecision + "\nweightedRecall: " +
      weightedRecall + "\naccuracy: " + accuracy + "\nf1: " + f1 + "\n";

    System.out.println(s);

    String outputFile = getOutputFile();
    Path filePath = Paths.get(outputFile);
    Files.deleteIfExists(filePath);
    byte[] strToBytes = s.getBytes();
    System.out.println("writing to file: " + outputFile);
    Files.write(filePath, strToBytes);
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

  private String getOutputFile() {
    return this.prop.getProperty("app.output.file");
  }

  private String getMasterNode() {
    return this.prop.getProperty("app.master.node");
  }

  private String getAccessKey() {
    return StringEscapeUtils.escapeJava(prop.getProperty("app.aws.access.key"));
  }

  private String getSecretKey() {
    return StringEscapeUtils.escapeJava(this.prop.getProperty("app.aws.secret.key"));
  }

  private String getSessionToken() {
    return StringEscapeUtils.escapeJava(this.prop.getProperty("app.aws.session.key"));
  }
}
