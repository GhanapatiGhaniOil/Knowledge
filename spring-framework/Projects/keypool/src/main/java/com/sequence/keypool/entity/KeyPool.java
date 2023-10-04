package com.sequence.keypool.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "KEYPOOL_10")
public class KeyPool {
	
	@Id
	@GeneratedValue(strategy = GenerationType.SEQUENCE)
	private Long keyId;

}
