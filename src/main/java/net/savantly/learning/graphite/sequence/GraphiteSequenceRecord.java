package net.savantly.learning.graphite.sequence;

import java.util.ArrayList;
import java.util.List;

import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;

public class GraphiteSequenceRecord implements SequenceRecord {
	
	private List<List<Writable>> record = new ArrayList<List<Writable>>();
	
	public GraphiteSequenceRecord() {}

	public GraphiteSequenceRecord(INDArray array) {
		for(int i = 0; i < array.rows(); i++) {
			INDArray row = array.getRow(i);
			List<Writable> featureList = new ArrayList<>();
			for (int j = 0; j < row.columns(); j++) {
				Writable writable = new DoubleWritable(row.getDouble(j));
				featureList.add(writable);
			}
			this.record.add(featureList);				
		}
		
	}

	@Override
	public List<List<Writable>> getSequenceRecord() {
		return record;
	}

	@Override
	public int getSequenceLength() {
		return record.size();
	}

	@Override
	public List<Writable> getTimeStep(int timeStep) {
		return this.record.get(timeStep);
	}

	@Override
	public void setSequenceRecord(List<List<Writable>> sequenceRecord) {
		this.record = sequenceRecord;
	}

	@Override
	public RecordMetaData getMetaData() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setMetaData(RecordMetaData recordMetaData) {
		// TODO Auto-generated method stub
		
	}

}
