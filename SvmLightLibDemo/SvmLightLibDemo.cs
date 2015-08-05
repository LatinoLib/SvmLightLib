/*==========================================================================;
 *
 *  This file is part of LATINO. See http://latino.sf.net
 *
 *  File:    SvmLightLibDemo.cs
 *  Desc:	 SVM^light and SVM^multiclass C# demo
 *  Created: Apr-2009
 * 
 *  Author:  Miha Grcar 
 * 
 ***************************************************************************/

using System;
using System.IO;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Globalization;
using System.Diagnostics;

namespace SvmLightLib.Demo
{
    /* .-----------------------------------------------------------------------
       |		 
       |  Class SvmLightLibDemo 
       |
       '-----------------------------------------------------------------------
    */
    public static class SvmLightLibDemo
    {
        private static List<byte> mBuffer
            = new List<byte>();
        private static int mIdx = 0;

        private static void Write(byte b)
        {
            mBuffer.Add(b);
        }

        private static byte Read()
        {
            return mBuffer[mIdx++];
        }

        private static int[] ReadFeatureVectors(StreamReader reader)
        {
            string line;
            List<int> feature_vectors = new List<int>();
            while ((line = reader.ReadLine()) != null)
            {
                if (!line.StartsWith("#"))
                {
                    Match label_match = new Regex(@"^(?<label>[+-]?\d+([.]\d+)?)(\s|$)").Match(line);
                    Debug.Assert(label_match.Success);
                    double label = Convert.ToDouble(label_match.Result("${label}"));
                    Match match = new Regex(@"(?<feature>(\d+|qid)):(?<weight>[-]?[\d\.]+)", RegexOptions.IgnoreCase).Match(line);                    
                    List<int> features = new List<int>();
                    List<float> weights = new List<float>();
                    int queryId = 0;
                    while (match.Success)
                    {
                        string featureStr = match.Result("${feature}");
                        if (featureStr.ToLower() != "qid")
                        {
                            int feature = Convert.ToInt32(featureStr);
                            float weight = Convert.ToSingle(match.Result("${weight}"), CultureInfo.InvariantCulture);
                            features.Add(feature);
                            weights.Add(weight);
                        }
                        else 
                        { 
                            queryId = Convert.ToInt32(match.Result("${weight}"), CultureInfo.InvariantCulture); 
                        }
                        match = match.NextMatch();
                    }
                    int vec_id = SvmLightLib.NewFeatureVector(features.Count, features.ToArray(), weights.ToArray(), label, queryId);
                    feature_vectors.Add(vec_id);
                }
            }
            return feature_vectors.ToArray();
        }

        public static void Main(string[] args)
        {
            // *** Test SVM^light induction ***

            Console.WriteLine("Testing SVM^light induction (API) ...");
            Console.WriteLine("Training ...");
            int[] train_set;
            using (StreamReader reader = new StreamReader(@"..\..\Examples\Inductive\train.dat"))
            {
                train_set = ReadFeatureVectors(reader);
            }
            int model_id = SvmLightLib.TrainModel("", train_set.Length, train_set);
            // test read/write 
            SvmLightLib.SaveModelBin(model_id, "model");
            SvmLightLib.DeleteModel(model_id);
            model_id = SvmLightLib.LoadModelBin("model");
            // test read/write callbacks
            SvmLightLib.WriteByteCallback wb = new SvmLightLib.WriteByteCallback(Write);
            SvmLightLib.SaveModelBinCallback(model_id, wb);
            GC.KeepAlive(wb);
            SvmLightLib.DeleteModel(model_id);
            SvmLightLib.ReadByteCallback rb = new SvmLightLib.ReadByteCallback(Read);
            model_id = SvmLightLib.LoadModelBinCallback(rb);
            GC.KeepAlive(rb);
            Console.WriteLine("Classifying ...");
            int[] test_set;
            using (StreamReader reader = new StreamReader(@"..\..\Examples\Inductive\test.dat"))
            {
                test_set = ReadFeatureVectors(reader);
            }
            int correct = 0;
            foreach (int vec_id in test_set)
            {
                int true_lbl = (int)SvmLightLib.GetFeatureVectorLabel(vec_id);
                SvmLightLib.Classify(model_id, 1, new int[] { vec_id });
                Debug.Assert(SvmLightLib.GetFeatureVectorClassifScoreCount(vec_id) == 1);
                double result = SvmLightLib.GetFeatureVectorClassifScore(vec_id, 0);
                int predicted_lbl = result > 0 ? 1 : -1;
                if (true_lbl == predicted_lbl) { correct++; }
            }
            Console.WriteLine("Accuracy: {0:0.00}%", (double)correct / (double)test_set.Length * 100.0);
            Console.WriteLine("CHECK: Expected accuracy: 97.67%");
            // cleanup
            SvmLightLib.DeleteModel(model_id);
            foreach (int[] arr in new int[][] { train_set, test_set }) foreach (int vec_id in arr)
            {
                SvmLightLib.DeleteFeatureVector(vec_id);
            }
            
            // *** Test SVM^light regression ***

            Console.WriteLine("Testing SVM^light regression (API) ...");
            Console.WriteLine("Training ...");
            using (StreamReader reader = new StreamReader(@"..\..\Examples\Regression\train.dat"))
            {
                train_set = ReadFeatureVectors(reader);
            }
            model_id = SvmLightLib.TrainModel("-z r", train_set.Length, train_set);
            // test read/write 
            SvmLightLib.SaveModelBin(model_id, "model");
            SvmLightLib.DeleteModel(model_id);
            model_id = SvmLightLib.LoadModelBin("model");
            // test read/write callbacks
            wb = new SvmLightLib.WriteByteCallback(Write);
            SvmLightLib.SaveModelBinCallback(model_id, wb);
            GC.KeepAlive(wb);
            SvmLightLib.DeleteModel(model_id);
            rb = new SvmLightLib.ReadByteCallback(Read);
            model_id = SvmLightLib.LoadModelBinCallback(rb);
            GC.KeepAlive(rb);
            Console.WriteLine("Classifying ...");
            using (StreamReader reader = new StreamReader(@"..\..\Examples\Regression\test.dat"))
            {
                test_set = ReadFeatureVectors(reader);
            }
            double mae = 0;
            foreach (int vec_id in test_set)
            {
                double true_lbl = SvmLightLib.GetFeatureVectorLabel(vec_id);
                SvmLightLib.Classify(model_id, 1, new int[] { vec_id });
                Debug.Assert(SvmLightLib.GetFeatureVectorClassifScoreCount(vec_id) == 1);
                double result = SvmLightLib.GetFeatureVectorClassifScore(vec_id, 0);
                mae += Math.Abs(true_lbl - result);
            }
            Console.WriteLine("MAE: {0:0.00}", mae / (double)test_set.Length);
            Console.WriteLine("CHECK: Expected MAE: 27.32");
            // cleanup
            SvmLightLib.DeleteModel(model_id);
            foreach (int[] arr in new int[][] { train_set, test_set }) foreach (int vec_id in arr)
            {
                SvmLightLib.DeleteFeatureVector(vec_id);
            }

            // *** Test SVM^light transduction ***

            Console.WriteLine("Testing SVM^light transduction (API) ...");
            Console.WriteLine("Training ...");
            using (StreamReader reader = new StreamReader(@"..\..\Examples\Transductive\train_transduction.dat"))
            {
                train_set = ReadFeatureVectors(reader);
            }
            model_id = SvmLightLib.TrainModel("", train_set.Length, train_set);
            // test read/write 
            SvmLightLib.SaveModelBin(model_id, "model");
            SvmLightLib.DeleteModel(model_id);
            model_id = SvmLightLib.LoadModelBin("model");
            // test read/write callbacks
            wb = new SvmLightLib.WriteByteCallback(Write);
            SvmLightLib.SaveModelBinCallback(model_id, wb);
            GC.KeepAlive(wb);
            SvmLightLib.DeleteModel(model_id);
            rb = new SvmLightLib.ReadByteCallback(Read);
            model_id = SvmLightLib.LoadModelBinCallback(rb);
            GC.KeepAlive(rb);
            Console.WriteLine("Classifying ...");
            using (StreamReader reader = new StreamReader(@"..\..\Examples\Transductive\test.dat"))
            {
                test_set = ReadFeatureVectors(reader);
            }
            correct = 0;
            foreach (int vec_id in test_set)
            {
                int true_lbl = (int)SvmLightLib.GetFeatureVectorLabel(vec_id);
                SvmLightLib.Classify(model_id, 1, new int[] { vec_id });
                Debug.Assert(SvmLightLib.GetFeatureVectorClassifScoreCount(vec_id) == 1);
                double result = SvmLightLib.GetFeatureVectorClassifScore(vec_id, 0);
                int predicted_lbl = result > 0 ? 1 : -1;
                if (true_lbl == predicted_lbl) { correct++; }
            }
            Console.WriteLine("Accuracy: {0:0.00}%", (double)correct / (double)test_set.Length * 100.0);
            Console.WriteLine("CHECK: Expected accuracy: 96.00%");
            // cleanup
            SvmLightLib.DeleteModel(model_id);
            foreach (int[] arr in new int[][] { train_set, test_set }) foreach (int vec_id in arr)
            {
                SvmLightLib.DeleteFeatureVector(vec_id);
            }

            // *** Test SVM^light preference ranking ***
        
            Console.WriteLine("Testing SVM^light preference ranking (API) ...");
            Console.WriteLine("Training ...");
            using (StreamReader reader = new StreamReader(@"..\..\Examples\Ranking\train.dat")) 
            { 
                train_set = ReadFeatureVectors(reader); 
            }
            model_id = SvmLightLib.TrainModel("-z p", train_set.Length, train_set);
            // test read/write 
            SvmLightLib.SaveModelBin(model_id, "model");
            SvmLightLib.DeleteModel(model_id);
            model_id = SvmLightLib.LoadModelBin("model");
            // test read/write callbacks
            wb = new SvmLightLib.WriteByteCallback(Write);
            SvmLightLib.SaveModelBinCallback(model_id, wb);
            GC.KeepAlive(wb);
            SvmLightLib.DeleteModel(model_id);
            rb = new SvmLightLib.ReadByteCallback(Read);
            model_id = SvmLightLib.LoadModelBinCallback(rb);
            GC.KeepAlive(rb);
            Console.WriteLine("Classifying ...");
            using (StreamReader reader = new StreamReader(@"..\..\Examples\Ranking\test.dat"))
            {
                test_set = ReadFeatureVectors(reader);
            }
            correct = 0;
            foreach (int vec_id in test_set)
            {
                SvmLightLib.Classify(model_id, 1, new int[] { vec_id });
                Debug.Assert(SvmLightLib.GetFeatureVectorClassifScoreCount(vec_id) == 1);
                double result = SvmLightLib.GetFeatureVectorClassifScore(vec_id, 0);
                Console.Write(result.ToString("0.00 "));
            }
            Console.WriteLine();
            Console.WriteLine("CHECK: Expected: 1.72 0.76 0.69 -0.49");
            // cleanup
            SvmLightLib.DeleteModel(model_id);
            foreach (int[] arr in new int[][] { train_set, test_set }) foreach (int vec_id in arr)
            {
                SvmLightLib.DeleteFeatureVector(vec_id);
            }

            // *** Test SVM^multiclass ***
            
            Console.WriteLine("Testing SVM^multiclass (command-line) ...");
            Console.WriteLine("Training ...");
            SvmLightLib._TrainMulticlassModel(@"-c 5000 ..\..\Examples\Multiclass\train.dat model");
            Console.WriteLine("Classifying ...");
            SvmLightLib._MulticlassClassify(@"..\..\Examples\Multiclass\test.dat model ..\..\Examples\Multiclass\out.dat");
            Console.WriteLine("CHECK: Expected zero/one-error on test set: 32.80%");
            Console.WriteLine("Testing SVM^multiclass (API) ...");
            using (StreamReader reader = new StreamReader(@"..\..\Examples\Multiclass\train.dat"))
            {
                train_set = ReadFeatureVectors(reader);
            }
            model_id = SvmLightLib.TrainMulticlassModel("-c 5000", train_set.Length, train_set);
            // test read/write 
            SvmLightLib.SaveMulticlassModelBin(model_id, "model");
            SvmLightLib.DeleteMulticlassModel(model_id);
            model_id = SvmLightLib.LoadMulticlassModelBin("model");
            // test read/write callbacks
            wb = new SvmLightLib.WriteByteCallback(Write);
            SvmLightLib.SaveMulticlassModelBinCallback(model_id, wb);
            GC.KeepAlive(wb);
            SvmLightLib.DeleteMulticlassModel(model_id);
            rb = new SvmLightLib.ReadByteCallback(Read);
            model_id = SvmLightLib.LoadMulticlassModelBinCallback(rb);
            GC.KeepAlive(rb);
            Console.WriteLine("Classifying ...");
            using (StreamReader reader = new StreamReader(@"..\..\Examples\Multiclass\test.dat"))
            {
                test_set = ReadFeatureVectors(reader);
            }
            correct = 0;
            foreach (int vec_id in test_set)
            {
                int true_lbl = (int)SvmLightLib.GetFeatureVectorLabel(vec_id);
                SvmLightLib.MulticlassClassify(model_id, 1, new int[] { vec_id });
                int n = SvmLightLib.GetFeatureVectorClassifScoreCount(vec_id);
                double max_score = double.MinValue;
                int predicted_lbl = -1;
                for (int i = 0; i < n; i++)
                {
                    double score = SvmLightLib.GetFeatureVectorClassifScore(vec_id, i);
                    if (score > max_score) { max_score = score; predicted_lbl = i + 1; }
                }
                if (true_lbl == predicted_lbl) { correct++; }
            }
            Console.WriteLine("Accuracy: {0:0.00}%", (double)correct / (double)test_set.Length * 100.0);
            Console.WriteLine("CHECK: Expected accuracy: 67.20%");
            // cleanup
            SvmLightLib.DeleteMulticlassModel(model_id);
            foreach (int[] arr in new int[][] { train_set, test_set }) foreach (int vec_id in arr)
            {
                SvmLightLib.DeleteFeatureVector(vec_id);
            }
        }
    }
}
