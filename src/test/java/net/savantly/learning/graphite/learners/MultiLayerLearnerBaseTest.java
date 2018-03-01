package net.savantly.learning.graphite.learners;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import net.savantly.learning.graphite.learners.MultiLayerLearnerBase;

public class MultiLayerLearnerBaseTest {

	@Test
	public void testLearner() {
		ExampleLearner learner = new ExampleLearner();
		learner.train();
	}

	class ExampleLearner extends MultiLayerLearnerBase {

		int numLinesToSkip = 0;
		char delimiter = ',';
		int labelIndex = 14;
		int numClasses = 2;
		int batchSize = 1000;
		
		@Override
		public int getFeatureCount() {
			return 14;
		}
		
		@Override
		public double getLearningRate() {
			return 0.5/14;
		}

		@Override
		public DataSetIterator getTestingDataSets() {
			try {
				RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
				recordReader.initialize(
						new FileSplit(new ClassPathResource("./data/training/eeg/eye_state.test.csv").getFile()));
				DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex,
						numClasses);
				return iterator;
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
		}

		@Override
		public DataSetIterator getTrainingDataSets() {
			try {
				RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
				recordReader.initialize(
						new FileSplit(new ClassPathResource("./data/training/eeg/eye_state.train.csv").getFile()));
				DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex,
						numClasses);
				return iterator;
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
		}

		@Override
		public int getNumberOfPossibleLabels() {
			return 2;
		}

		@Override
		public int getNumberOfIterations() {
			return 40;
		}

		@Override
		public DataNormalization getNormalizer() {
			return new NormalizerStandardize();
		}

	}

}
