package com.sequence.keypool.entity;

import java.util.Date;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.NamedQuery;
import jakarta.persistence.Table;
import lombok.Data;

@Data
@Entity
@Table(name = "KEYPOOL_10")
@NamedQuery
public class KeyPool {
	
	private static final long serialVersionUID = -2952735933715107252L;
	
	@Id
	@GeneratedValue(strategy = GenerationType.SEQUENCE)
	@Column(name = "keyId")
	private Long keyId;
	
	
	@Column(name = "source")
	private String source;
	
	@Column(name="crtTs")	
	private Date crtTs;

}
