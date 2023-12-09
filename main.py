from sentimentAnalysis import logger
from sentimentAnalysis.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from sentimentAnalysis.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from sentimentAnalysis.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from sentimentAnalysis.pipeline.stage_04_model_training import ModelTrainerTrainingPipeline
from sentimentAnalysis.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline

if __name__ == '__main__':
   
   STAGE_NAME = "Data Ingestion stage"
   try:
      logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
      data_ingestion = DataIngestionTrainingPipeline()
      data_ingestion.main()
      logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
   except Exception as e:
         logger.exception(e)
         raise e


   STAGE_NAME = "Data Validation stage"
   try:
      logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
      data_ingestion = DataValidationTrainingPipeline()
      data_ingestion.main()
      logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
   except Exception as e:
         logger.exception(e)
         raise e

   STAGE_NAME = "Data Transformation stage"
   try:
      logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
      data_ingestion = DataTransformationTrainingPipeline()
      data_ingestion.main()
      logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
   except Exception as e:
         logger.exception(e)
         raise e
   
   STAGE_NAME = "Model Training stage"
   try:
      logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
      data_ingestion = ModelTrainerTrainingPipeline()
      data_ingestion.main()
      logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
   except Exception as e:
         logger.exception(e)
         raise e
   
   STAGE_NAME = "Model Evaluation stage"
   try:
      logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
      data_ingestion = ModelEvaluationTrainingPipeline()
      data_ingestion.main()
      logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
   except Exception as e:
         logger.exception(e)
         raise e