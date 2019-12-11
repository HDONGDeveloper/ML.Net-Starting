using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace SentimentAnalysis
{
    /// <summary>
    /// 情感类
    /// LoadColumn 用来做数据文件排序
    /// ColumnName 加入标签（todo 感觉是最终返回数的意思）
    /// </summary>
    public class SentimentData
    {
        /// <summary>
        /// 说出的话
        /// </summary>
        [LoadColumn(0)]
        public string SentimentText;

        /// <summary>
        /// 情绪的表现形式 正面 还是 负面
        /// </summary>
        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment;
    }

    /// <summary>
    /// 预测类
    /// </summary>
    public class SentimentPrediction : SentimentData
    {
        /// <summary>
        /// 预测结果
        /// </summary>

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        /// <summary>
        /// 概率
        /// </summary>
        public float Probability { get; set; }
        /// <summary>
        /// 模型得分
        /// </summary>
        public float Score { get; set; }
    }
}
