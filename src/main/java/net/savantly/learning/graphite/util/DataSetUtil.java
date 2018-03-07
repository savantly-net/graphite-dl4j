package net.savantly.learning.graphite.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.datavec.api.writable.FloatWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;

public class DataSetUtil {
	
	public static SplitTestAndTrain splitTestAndTrain3d(DataSet ds, double percentTrain) {
		int seriesLength = ds.getFeatures().size(0);
		int featureLength = ds.getFeatures().size(1);
		int timeSteps = ds.getFeatures().size(2);
		int numSamples = (int)Math.round(timeSteps * percentTrain);
		
		List<Integer> randomInts = new ArrayList<>(timeSteps);
		for(int i=0; i<timeSteps; i++) {
			randomInts.add(i);
		}
		Collections.shuffle(randomInts);
		INDArray trainFeatures = Nd4j.create(new int[] {seriesLength, featureLength, numSamples});
		INDArray trainLabels = Nd4j.create(new int[] {seriesLength, featureLength, numSamples});
		INDArray testFeatures = Nd4j.create(new int[] {seriesLength, featureLength, timeSteps-numSamples});
		INDArray testLabels = Nd4j.create(new int[] {seriesLength, featureLength, timeSteps-numSamples});
		// TODO: INDArray mask = Nd4j.create(new int[] {seriesLength, featureLength, timeSteps});
		for (int i=0; i<seriesLength; i++) {
			for (int j=0; j<featureLength; j++) {
				for (int k=0; k<timeSteps; k++) {
					if (k < numSamples) {
						trainFeatures.put(new int[] {i,j,k}, ds.getFeatures().getScalar(i,j,k));
						trainLabels.put(new int[] {i,j,k}, ds.getLabels().getScalar(i,j,k));
					} else {
						testFeatures.put(new int[] {i,j,k-numSamples}, ds.getFeatures().getScalar(i,j,k));
						testLabels.put(new int[] {i,j,k-numSamples}, ds.getLabels().getScalar(i,j,k));
					}
				}
			}
		}
		DataSet train = new DataSet(trainFeatures, trainLabels);
		DataSet test = new DataSet(testFeatures, testLabels);
		
		SplitTestAndTrain split = new SplitTestAndTrain(train, test);
		return split;
	}
	
	public static SplitTestAndTrain splitTestAndTrain(DataSet ds, double percentTrain) {
		if (ds.getFeatures().rank() == 2) {
			return ds.splitTestAndTrain(percentTrain);
		} else {
			return splitTestAndTrain3d(ds, percentTrain);
		}
	}
	

	public static List<List<List<Writable>>> convert3DToWritableSequence(INDArray ndArray) {
		if (ndArray.rank() != 3) {
			throw new RuntimeException("the data must be 3d");
		}
		int[] shape = ndArray.shape();
		int sequenceGroups = shape[0];
		int featureCount = shape[1];
		int timeSteps = shape[2];
		
		List<List<List<Writable>>> sequenceList = new ArrayList<List<List<Writable>>>();
		
		AtomicInteger sequenceCounter = new AtomicInteger(0);
		while (sequenceCounter.get() < sequenceGroups) {
			List<List<Writable>> timeStepList = new ArrayList<>();
			AtomicInteger timeStepCounter = new AtomicInteger(0);
			while(timeStepCounter.get() < timeSteps) {
				List<Writable> featureList = new ArrayList<>();
				AtomicInteger featureCounter = new AtomicInteger(0);
				while (featureCounter.get() < featureCount) {
					featureList.add(new 
							FloatWritable(ndArray.getFloat(new int[] {sequenceCounter.get(), featureCounter.get(), timeStepCounter.get()})));
					featureCounter.incrementAndGet();
				}
				timeStepList.add(featureList);
				timeStepCounter.incrementAndGet();
			}
			sequenceList.add(timeStepList);
			sequenceCounter.incrementAndGet();
		}
		return sequenceList;
	}
	

	public static List<List<List<Writable>>> convert2DToWritableSequence(INDArray ndArray) {
		if (ndArray.rank() != 2) {
			throw new RuntimeException("the data must be 2d");
		}
		int[] shape = ndArray.shape();
		int dim0 = 1;
		int dim1 = shape[0];
		int dim2 = shape[1];
		
		List<List<List<Writable>>> writables = new ArrayList<List<List<Writable>>>();
		
		AtomicInteger counterDim0 = new AtomicInteger(0);
		while (counterDim0.get() < dim0) {
			List<List<Writable>> writablesDim1 = new ArrayList<>();
			AtomicInteger counterDim1 = new AtomicInteger(0);
			while(counterDim1.get() < dim1) {
				List<Writable> writablesDim2 = new ArrayList<>();
				AtomicInteger counterDim2 = new AtomicInteger(0);
				while (counterDim2.get() < dim2) {
					writablesDim2.add(new 
							FloatWritable(ndArray.getFloat(new int[] {counterDim1.get(), counterDim2.get()})));
					counterDim2.incrementAndGet();
				}
				writablesDim1.add(writablesDim2);
				counterDim1.incrementAndGet();
			}
			writables.add(writablesDim1);
			counterDim0.incrementAndGet();
		}
		return writables;
	}
	
	public static List<List<List<Writable>>> convertToWritableSequence(INDArray ndArray) {
		if (ndArray.rank() == 2) {
			return convert2DToWritableSequence(ndArray);
		} else {
			return DataSetUtil.convert3DToWritableSequence(ndArray);
		}
	}

}
