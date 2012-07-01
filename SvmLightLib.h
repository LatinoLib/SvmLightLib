/*==========================================================================;
 *
 *  This file is part of LATINO. See http://latino.sf.net
 *
 *  File:    SvmLightLib.h
 *  Desc:	 SVM^light and SVM^multiclass DLL wrapper
 *  Created: Aug-2007
 *
 *  Author:  Miha Grcar 
 *
 *  This software is available for non-commercial use only. It must not be 
 *  modified and distributed without prior permission of the author of 
 *  SVM^light and SVM^struct (Thorsten Joachims). None of the authors is
 *  responsible for implications from the use of this software.                                  
 * 
 ***************************************************************************/

#ifndef SVMLIGHTLIB_H
#define SVMLIGHTLIB_H

#ifdef SVMLIGHTLIB_EXPORTS
#define SVMLIGHTLIB_API extern "C" __declspec(dllexport)
#else
#define SVMLIGHTLIB_API extern "C" __declspec(dllimport)
#endif

typedef void (__stdcall *WriteByteCallback)(char byte);
typedef char (__stdcall *ReadByteCallback)();

// label is 1 or -1 for inductive binary SVM; 1, -1, or 0 (unlabeled) for transductive binary SVM; 
// positive integer for multiclass SVM; real value for SVM regression
SVMLIGHTLIB_API int NewFeatureVector(int feature_count, int *features, float *weights, double label);
SVMLIGHTLIB_API void DeleteFeatureVector(int id);
SVMLIGHTLIB_API int GetFeatureVectorFeatureCount(int feature_vector_id);
SVMLIGHTLIB_API int GetFeatureVectorFeature(int feature_vector_id, int feature_idx);
SVMLIGHTLIB_API float GetFeatureVectorWeight(int feature_vector_id, int feature_idx);
SVMLIGHTLIB_API double GetFeatureVectorLabel(int feature_vector_id);
SVMLIGHTLIB_API void SetFeatureVectorLabel(int feature_vector_id, double label);
SVMLIGHTLIB_API int GetFeatureVectorClassifScoreCount(int feature_vector_id);
SVMLIGHTLIB_API double GetFeatureVectorClassifScore(int feature_vector_id, int classif_score_idx);

SVMLIGHTLIB_API void _TrainModel(char *args);
SVMLIGHTLIB_API int TrainModel(char *args, int feature_vector_count, int *feature_vectors);
SVMLIGHTLIB_API void SaveModel(int model_id, char *file_name);
SVMLIGHTLIB_API int LoadModel(char *file_name);
SVMLIGHTLIB_API void SaveModelBin(int model_id, char *file_name);
SVMLIGHTLIB_API int LoadModelBin(char *file_name);
SVMLIGHTLIB_API void SaveModelBinCallback(int model_id, WriteByteCallback callback);
SVMLIGHTLIB_API int LoadModelBinCallback(ReadByteCallback callback);
SVMLIGHTLIB_API void _Classify(char *args);
SVMLIGHTLIB_API void Classify(int model_id, int feature_vector_count, int *feature_vectors);
SVMLIGHTLIB_API void DeleteModel(int id);
SVMLIGHTLIB_API double GetHyperplaneBias(int model_id);
SVMLIGHTLIB_API int GetSupportVectorCount(int model_id);
SVMLIGHTLIB_API int GetSupportVectorFeatureCount(int model_id, int sup_vec_idx);
SVMLIGHTLIB_API int GetSupportVectorFeature(int model_id, int sup_vec_idx, int feature_idx);
SVMLIGHTLIB_API float GetSupportVectorWeight(int model_id, int sup_vec_idx, int feature_idx);
SVMLIGHTLIB_API double GetSupportVectorAlpha(int model_id, int sup_vec_idx);
SVMLIGHTLIB_API int GetSupportVectorIndex(int model_id, int sup_vec_idx);
SVMLIGHTLIB_API int GetKernelType(int model_id);
SVMLIGHTLIB_API int GetFeatureCount(int model_id);
SVMLIGHTLIB_API double GetLinearWeight(int model_id, int feature_idx);

SVMLIGHTLIB_API void _TrainMulticlassModel(char *args);
SVMLIGHTLIB_API int TrainMulticlassModel(char *args, int feature_vector_count, int *feature_vectors);
SVMLIGHTLIB_API void SaveMulticlassModel(int model_id, char *file_name);
SVMLIGHTLIB_API int LoadMulticlassModel(char *file_name);
SVMLIGHTLIB_API void SaveMulticlassModelBin(int model_id, char *file_name);
SVMLIGHTLIB_API int LoadMulticlassModelBin(char *file_name);
SVMLIGHTLIB_API void SaveMulticlassModelBinCallback(int model_id, WriteByteCallback callback);
SVMLIGHTLIB_API int LoadMulticlassModelBinCallback(ReadByteCallback callback);
SVMLIGHTLIB_API void _MulticlassClassify(char *args);
SVMLIGHTLIB_API void MulticlassClassify(int model_id, int feature_vector_count, int *feature_vectors);
SVMLIGHTLIB_API void DeleteMulticlassModel(int id);

#endif