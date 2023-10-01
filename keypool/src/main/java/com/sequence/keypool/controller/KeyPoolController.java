package com.sequence.keypool.controller;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import com.sequence.keypool.service.KeyPoolService;

@RestController
public class KeyPoolController {
	
	private Logger log = LoggerFactory.getLogger(KeyPoolController.class); 

	@Autowired
	private KeyPoolService keyService;

	@GetMapping("/seq")
	public Long nextSequence() {
		log.info("Invoked /seq endpoint");
		keyService.nextSequence();
		
		return null;

	}

	@GetMapping("/rsd")
	public void reserved() {
		keyService.reserved();

	}

	@GetMapping("/list/seq")
	public List<Long> listSequence() {
		keyService.listSequence();
		
		return null;
	}

	@GetMapping("/list/rsd")
	public List<Long> listReserved() {
		keyService.listReserved();
		return null;
	}

}
