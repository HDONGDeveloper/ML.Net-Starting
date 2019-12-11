using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;

namespace SentimentAnalysis
{
    class Program
    {
        // <SnippetDeclareGlobalVariables> 
        // 声明数据级目录
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "yelp_labelled.txt");
        // </SnippetDeclareGlobalVariables>

        static void Main(string[] args)
        {
            // Create ML.NET context/local environment - allows you to add steps in order to keep everything together 
            // as you discover the ML.NET trainers and transforms .
            // 创建一个识别的环境 
            // <SnippetCreateMLContext>
            MLContext mlContext = new MLContext();
            // </SnippetCreateMLContext>

            // <SnippetCallLoadData>
            TrainTestData splitDataView = LoadData(mlContext);
            // </SnippetCallLoadData>


            // <SnippetCallBuildAndTrainModel>
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            // </SnippetCallBuildAndTrainModel>

            // <SnippetCallEvaluate>
            Evaluate(mlContext, model, splitDataView.TestSet);
            // </SnippetCallEvaluate>

            // <SnippetCallUseModelWithSingleItem>
            UseModelWithSingleItem(mlContext, model);
            // </SnippetCallUseModelWithSingleItem>

            // <SnippetCallUseModelWithBatchItems>
            UseModelWithBatchItems(mlContext, model);
            // </SnippetCallUseModelWithBatchItems>

            Console.WriteLine();
            Console.WriteLine("=============== End of process ===============");
        }

        /// <summary>
        /// 加载数据。
        /// 将加载的数据集拆分为训练数据集和测试数据集。
        /// 返回拆分的训练数据集和测试数据集。
        /// </summary>
        /// <param name="mlContext">环境</param>
        /// <returns></returns>
        public static TrainTestData LoadData(MLContext mlContext)
        {
            // Note that this case, loading your training data from a file, 
            // is the easiest way to get started, but ML.NET also allows you 
            // to load data from databases or in-memory collections.
            // 1.加载数据  读取目录中的数据集 传递文件路径 和 是否包含头 给出泛型的类
            // <SnippetLoadData>
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            // </SnippetLoadData>

            // You need both a training dataset to train the model and a test dataset to evaluate the model.
            // Split the loaded dataset into train and test datasets
            // Specify test dataset percentage with the `testFraction`parameter
            // 2. 讲加载的数据集拆分成训练集 和 测试数据集   传入要拆分的参数 和 测试数据比例
            // <SnippetSplitData>
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.1);
            // </SnippetSplitData>

            // <SnippetReturnSplitData>  
            // 3.返回数据集（训练集/测试集）
            return splitDataView;
            // </SnippetReturnSplitData>           
        }

        /// <summary>
        /// 提取并转换数据。
        /// 定型模型。
        /// 根据测试数据预测情绪。
        /// 返回模型。
        /// </summary>
        /// <param name="mlContext">环境</param>
        /// <param name="splitTrainSet">训练集</param>
        /// <returns></returns>
        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            // Create a flexible pipeline (composed by a chain of estimators) for creating/training the model.
            // This is used to format and clean the data.  
            // Convert the text column to numeric vectors (Features column) 
            // 1.提取并转换数据   新产生的特征列 ， 用那个列产生（如果为空读取实体outputColumnName标注）
            // <SnippetFeaturizeText>
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
            //</SnippetFeaturizeText>
            // append the machine learning task to the estimator
            // 2.定义使用的迅雷模型 为 二元分类（SdcaLogisticRegression）   
            // <SnippetAddTrainer> 
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
            // </SnippetAddTrainer>
            // Create and train the model based on the dataset that has been loaded, transformed.
            // 开始创建训练模型
            // <SnippetTrainModel>
            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();
            // </SnippetTrainModel>

            // Returns the model we trained to use for evaluation.
            // 返回训练好的模型
            // <SnippetReturnModel>
            return model;
            // </SnippetReturnModel>
        }


        /// <summary>
        /// 加载测试数据集。
        /// 创建 BinaryClassification 计算器。
        /// 评估模型并创建指标。
        /// 显示指标。
        /// </summary>
        /// <param name="mlContext">环境</param>
        /// <param name="model">训练好的模型</param>
        /// <param name="splitTestSet">训练集</param>
        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            // Evaluate the model and show accuracy stats
            //Take the data in, make transformations, output the data. 
            // 用测试集对模型进行评估/只是把测试集加载进去（懒加载的形式） 预测集
            // <SnippetTransformData>
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);
            // </SnippetTransformData>

            // BinaryClassificationContext.Evaluate returns a BinaryClassificationEvaluator.CalibratedResult
            // that contains the computed overall metrics.
            // 把预测集 扔进去 指定测试集中的lable 对进度进行评估 返回评估的打分
            // <SnippetEvaluate>
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            // </SnippetEvaluate>

            // The Accuracy metric gets the accuracy of a model, which is the proportion 
            // of correct predictions in the test set.

            // The AreaUnderROCCurve metric is equal to the probability that the algorithm ranks
            // a randomly chosen positive instance higher than a randomly chosen negative one
            // (assuming 'positive' ranks higher than 'negative').

            // The F1Score metric gets the model's F1 score.
            // The F1 score is the harmonic mean of precision and recall:
            //  2 * precision * recall / (precision + recall).

            // <SnippetDisplayMetrics>
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation/ 模型质量量度评估");
            Console.WriteLine("--------------------------------");
            //Accuracy 指标可获取模型的准确性，即测试集中正确预测所占的比例。
            Console.WriteLine($"Accuracy/正确比例: {metrics.Accuracy:P2}");
            //AreaUnderRocCurve 指标指示模型对正面类和负面类进行正确分类的置信度。 应该使 AreaUnderRocCurve 尽可能接近 1。
            Console.WriteLine($"Auc/todo鉴别: {metrics.AreaUnderRocCurve:P2}");
            //F1Score 指标可获取模型的 F1 分数，该分数是查准率和查全率之间的平衡关系的度量值。 应该使 F1Score 尽可能接近 1。
            Console.WriteLine($"F1Score/模型的F1得分: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
            //</SnippetDisplayMetrics>

        }

        /// <summary>
        /// 创建测试数据的单个注释。
        /// 根据测试数据预测情绪。
        /// 结合测试数据和预测进行报告。
        /// 显示预测结果。
        /// </summary>
        /// <param name="mlContext">环境</param>
        /// <param name="model">模型</param>
        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            // <SnippetCreatePredictionEngine1>
            // PredictionEngine 是一个简便 API，可使用它对单个数据实例执行预测   PredictionEngine 不是线程安全型
            // 生产环境请使用 PredictionEnginePool服务
            // 1. 创建一次性预测引擎
            // 2. 训练实体
            // 3. 测试实体 必须包含同样的列
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            // </SnippetCreatePredictionEngine1>

            // <SnippetCreateTestIssue1>
            // 创建一个测试对象 写入说出的话
            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "This was a very bad steak"
            };
            // </SnippetCreateTestIssue1>

            // <SnippetPredict>
            //将测试评论数据传递到 PredictionEngine
            //Predict() 函数对单行数据进行预测
            var resultPrediction = predictionFunction.Predict(sampleStatement);
            // </SnippetPredict>
            // <SnippetOutputPrediction>
            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
            // </SnippetOutputPrediction>
        }

        /// <summary>
        /// 创建批处理测试数据。
        /// 根据测试数据预测情绪。
        /// 结合测试数据和预测进行报告。
        /// 显示预测结果。
        /// </summary>
        /// <param name="mlContext">环境</param>
        /// <param name="model">模型</param>
        public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {
            // Adds some comments to test the trained model's data points.
            // 创建多个数据进行测试
            // <SnippetCreateTestIssues>
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This was a horrible meal"
                },
                new SentimentData
                {
                    SentimentText = "I love this spaghetti."
                }
            };
            // </SnippetCreateTestIssues>

            // Load batch comments just created 
            // 把对象变成数据集 懒加载
            // <SnippetPrediction>
            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);
            // 把数据集加载到识别模型中 懒加载
            IDataView predictions = model.Transform(batchComments);

            // Use model to predict whether comment data is Positive (1) or Negative (0).
            // 开始进行预测
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);
            // </SnippetPrediction>

            // <SnippetAddInfoMessage>
            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
            // </SnippetAddInfoMessage>

            Console.WriteLine();

            // <SnippetDisplayResults>
            // 对返回的数据进行输出
            foreach (SentimentPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");

            }
            Console.WriteLine("=============== End of predictions ===============");
            // </SnippetDisplayResults>       
        }

    }
}
