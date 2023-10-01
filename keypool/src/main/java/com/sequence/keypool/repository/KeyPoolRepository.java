package com.sequence.keypool.repository;

import com.sequence.keypool.entity.KeyPool;

interface KeyPoolRepository extends BaseRepository<KeyPool, Long> {
	Long keyId();
}
