package net.savantly.learning.graphite.sequence;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.listener.RecordListener;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class GraphiteSequenceRecordReader implements SequenceRecordReader {
	
	private static Logger log = LoggerFactory.getLogger(GraphiteSequenceRecordReader.class);
	
	private AtomicInteger sequenceCounter = new AtomicInteger(-1);
	private AtomicInteger timeStepCounter = new AtomicInteger(-1);
	private List<GraphiteSequenceRecord> sequenceRecords = new ArrayList<>();

	private List<RecordListener> recordListeners;

	public GraphiteSequenceRecordReader(List<INDArray> listOfFeaturesValuesPairs) {
		listOfFeaturesValuesPairs.stream().forEach(a -> {
			if(a.length() > 0) {
				this.sequenceRecords.add(new GraphiteSequenceRecord(a));
			} else {
				log.debug("empty sequence?");
				throw new RuntimeException("Empty sequence!");
			}
		});
	}

	@Override
	public void initialize(InputSplit split) throws IOException, InterruptedException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public boolean batchesSupported() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public List<Writable> next(int num) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<Writable> next() {
        return this.sequenceRecords.get(
        		this.sequenceCounter.get()).getSequenceRecord().get(
        				this.timeStepCounter.incrementAndGet());
	}
	
    boolean hasNextTimeStep() {
    	if (this.sequenceCounter.get() > -1) {
    		if (this.sequenceCounter.get() < this.sequenceRecords.size()) {
    			return this.timeStepCounter.get() < this.sequenceRecords.get(this.sequenceCounter.get()).getSequenceLength();
    		}
    	}
    	return false;
    };

	@Override
	public boolean hasNext() {
		
		return this.sequenceCounter.get()+1 < this.sequenceRecords.size();
		
/*		if (hasNextTimeStep()) {
            return true;
        }
        if (this.sequenceCounter.get() < this.sequenceRecords.size()) {
        	this.advanceToNextLocation();
            return true;
        }
        return (hasNextTimeStep());*/
	}

	@Override
	public List<String> getLabels() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void reset() {
		this.sequenceCounter.set(-1);
		this.timeStepCounter.set(-1);
	}

	@Override
	public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Record nextRecord() {
		return null;
	}

	@Override
	public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<RecordListener> getListeners() {
		return this.recordListeners;
	}

	@Override
	public void setListeners(RecordListener... listeners) {
		this.recordListeners = new ArrayList<RecordListener>(listeners.length);
		for (int i = 0; i < listeners.length; i++) {
			this.recordListeners.add(listeners[i]);
		}
	}

	@Override
	public void setListeners(Collection<RecordListener> listeners) {
		this.recordListeners = new ArrayList<>();
		this.recordListeners.addAll(listeners);
	}

	@Override
	public void close() throws IOException {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void setConf(Configuration conf) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Configuration getConf() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<List<Writable>> sequenceRecord() {
		return this.nextSequence().getSequenceRecord();
	}

	@Override
	public List<List<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
		return null;
	}

	@Override
	public GraphiteSequenceRecord nextSequence() {
		return this.sequenceRecords.get(this.sequenceCounter.incrementAndGet());
	}

	@Override
	public GraphiteSequenceRecord loadSequenceFromMetaData(RecordMetaData recordMetaData) throws IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<SequenceRecord> loadSequenceFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
		// TODO Auto-generated method stub
		return null;
	}

}
