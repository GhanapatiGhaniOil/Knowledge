package com.sequence.keypool;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.domain.EntityScan;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;

@SpringBootApplication
@EntityScan("com.sequence.keypool.entity")
@EnableJpaRepositories(basePackages = "com.sequence.keypool.repositories")
public class KeypoolApplication {

	public static void main(String[] args) {
		SpringApplication.run(KeypoolApplication.class, args);
	}

}
