package com.sequence.keypool.controller;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import com.sequence.keypool.service.KeyPoolService;

@RestController
public class KeyPoolController {
	
	@Autowired
	private KeyPoolService keyService;
	
	/*
	 * @GetMapping("/seq") public Long nextSequence() { keyService.nextSequence();
	 * 
	 * }
	 * 
	 * @GetMapping("/rsd") public void reserved() { keyService.reserved();
	 * 
	 * }
	 * 
	 * @GetMapping("/list/seq") public List<Long> listSequence() {
	 * keyService.listSequence();
	 * 
	 * }
	 * 
	 * @GetMapping("/list/rsd") public List<Long> listReserved() {
	 * keyService.listReserved();
	 * 
	 * }
	 */
	

}
