/*==========================================================================;
 *
 *  This file is part of LATINO. See http://latino.sf.net
 *
 *  File:    SvmLightLib.cs
 *  Desc:	 SVM^light and SVM^multiclass C# wrapper
 *  Created: Aug-2007
 * 
 *  Author:  Miha Grcar 
 * 
 ***************************************************************************/

using System.Runtime.InteropServices;

namespace SvmLightLib
{
    /* .-----------------------------------------------------------------------
       |		 
       |  Class SvmLightLib 
       |
       '-----------------------------------------------------------------------
    */
    internal static class SvmLightLib
    {
#if DEBUG
        const string SVMLIGHTLIB_DLL = "SvmLightLibDebug.dll";
#else
        const string SVMLIGHTLIB_DLL = "SvmLightLib.dll";
#endif
        public delegate void WriteByteCallback(byte b);
        public delegate byte ReadByteCallback();

        // label is 1 or -1 for inductive binary SVM; 1, -1, or 0 (unlabeled) for transductive binary SVM; 
        // a positive integer for multiclass SVM; a real value for SVM regression
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern int NewFeatureVector(int feature_count, int[] features, float[] weights, double label);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern void DeleteFeatureVector(int id);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern int GetFeatureVectorFeatureCount(int feature_vector_id);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern int GetFeatureVectorFeature(int feature_vector_id, int feature_idx);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern float GetFeatureVectorWeight(int feature_vector_id, int feature_idx);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern double GetFeatureVectorLabel(int feature_vector_id);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern void SetFeatureVectorLabel(int feature_vector_id, double label);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern int GetFeatureVectorClassifScoreCount(int feature_vector_id);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern double GetFeatureVectorClassifScore(int feature_vector_id, int classif_score_idx);

        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern void _TrainModel(string args);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern int TrainModel(string args, int feature_vector_count, int[] feature_vectors);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern void SaveModel(int model_id, string file_name);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern int LoadModel(string file_name);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern void SaveModelBin(int model_id, string file_name);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern int LoadModelBin(string file_name);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern void SaveModelBinCallback(int model_id, WriteByteCallback callback);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern int LoadModelBinCallback(ReadByteCallback callback);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern void _Classify(string args);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern void Classify(int model_id, int feature_vector_count, int[] feature_vectors);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern void DeleteModel(int id);

        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern void _TrainMulticlassModel(string args);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern int TrainMulticlassModel(string args, int feature_vector_count, int[] feature_vectors);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern void SaveMulticlassModel(int model_id, string file_name);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern int LoadMulticlassModel(string file_name);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern void SaveMulticlassModelBin(int model_id, string file_name);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern int LoadMulticlassModelBin(string file_name);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern void SaveMulticlassModelBinCallback(int model_id, WriteByteCallback callback);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern int LoadMulticlassModelBinCallback(ReadByteCallback callback);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern void _MulticlassClassify(string args);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern void MulticlassClassify(int model_id, int feature_vector_count, int[] feature_vectors);
        [DllImport(SVMLIGHTLIB_DLL)]
        public static extern void DeleteMulticlassModel(int id);
    }
}