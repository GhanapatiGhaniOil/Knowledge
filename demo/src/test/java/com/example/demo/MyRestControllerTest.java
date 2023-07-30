package com.example.demo;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.ResponseEntity;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
public class MyRestControllerTest {
	
	@Autowired
	private TestRestTemplate template;
	
	@Test
	public void getUser() throws Exception {
		ResponseEntity<String> response = template.getForEntity("/users/"+10034, String.class);
		assertThat(response.getBody()).isEqualTo("Greeting from user:"+10034);
	}
}
