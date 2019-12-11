﻿using Microsoft.ML.Data;
using Microsoft.ML;
using System;

namespace ML.NET_World
{
    class Program
    {
        public class HouseData
        {

            public float Size { get; set; }
            public float Price { get; set; }

        }

        public class Prediction
        {

            [ColumnName("Score")]
            public float Price { get; set; }
        }


        static void Main(string[] args)
        {
            MLContext mLContext = new MLContext();

            HouseData[] houseDatas = {
                new HouseData() { Size = 1.1f, Price = 1.2f },
                new HouseData() { Size = 1.9f, Price = 2.3f },
                new HouseData() { Size = 2.8f, Price = 3.0f },
                new HouseData() { Size = 3.4f, Price = 3.7f }
            };



            IDataView traningData = mLContext.Data.LoadFromEnumerable(houseDatas);


            var pipeline = mLContext.Transforms.Concatenate("Features", new[] { "Size" })
                .Append(mLContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 10000));

            var model = pipeline.Fit(traningData);

            //mLContext.Model.Save(model, traningData.Schema, "model.zip");

            var size = new HouseData() { Size = 2.5f };
            var price = mLContext.Model.CreatePredictionEngine<HouseData, Prediction>(model).Predict(size);

            Console.WriteLine($"Predicted price for size: {size.Size * 1000} sq ft= {price.Price * 100:C}k");


            HouseData[] testHouseData =
    {
            new HouseData() { Size = 1.1F, Price = 0.98F },
            new HouseData() { Size = 1.9F, Price = 2.1F },
            new HouseData() { Size = 2.8F, Price = 2.9F },
            new HouseData() { Size = 3.4F, Price = 3.6F }
        };

            var testHouseDataView = mLContext.Data.LoadFromEnumerable(testHouseData);
            var testPriceDataView = model.Transform(testHouseDataView);

            var debug = testPriceDataView.Preview();

            var metrics = mLContext.Regression.Evaluate(testPriceDataView, labelColumnName: "Price");

            Console.WriteLine($"R^2: {metrics.RSquared:0.##}");
            Console.WriteLine($"RMS error: {metrics.RootMeanSquaredError:0.##}");

            Console.ReadLine();

        }
    }
}
