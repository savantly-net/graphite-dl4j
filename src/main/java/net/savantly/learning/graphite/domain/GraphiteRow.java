package net.savantly.learning.graphite.domain;


import org.joda.time.DateTime;

public class GraphiteRow implements Comparable<GraphiteRow> {
	
	private String target;
	private float value;
	private DateTime epoch;
	
	public GraphiteRow(String target, float value, DateTime epoch) {
		this.target = target;
		this.value = value;
		this.epoch = epoch;
	}
	
	public GraphiteRow(String target, String value, String epoch) {
		this.target = target;
		this.value = Float.parseFloat(value);
		this.epoch = DateTime.parse(epoch.replace(" ", "T"));
	}

	public GraphiteRow(String target, Float value, String epoch) {
		this.target = target;
		this.value = value;
		this.epoch = DateTime.parse(epoch.replace(" ", "T"));
	}

	public String getTarget() {
		return target;
	}
	public float getValue() {
		return value;
	}
	public DateTime getEpoch() {
		return epoch;
	}

	@Override
	public int compareTo(GraphiteRow o) {
		return this.getEpoch().compareTo(o.getEpoch());
	}

}
