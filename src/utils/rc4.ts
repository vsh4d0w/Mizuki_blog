/**
 * RC4 加密/解密算法实现
 * RC4 是一种流加密算法,加密和解密使用相同的函数
 */

/**
 * 初始化 RC4 密钥调度算法 (KSA)
 * @param key - 密钥字符串或字节数组
 * @returns S-Box 数组
 */
function initRC4(key: string | Uint8Array): Uint8Array {
  const keyBytes = typeof key === 'string' 
    ? new TextEncoder().encode(key) 
    : key;
  
  const keyLength = keyBytes.length;
  const S = new Uint8Array(256);
  
  // 初始化 S-Box
  for (let i = 0; i < 256; i++) {
    S[i] = i;
  }
  
  // 密钥调度算法
  let j = 0;
  for (let i = 0; i < 256; i++) {
    j = (j + S[i] + keyBytes[i % keyLength]) % 256;
    // 交换 S[i] 和 S[j]
    [S[i], S[j]] = [S[j], S[i]];
  }
  
  return S;
}

/**
 * RC4 伪随机生成算法 (PRGA)
 * @param S - S-Box 数组
 * @param data - 要加密/解密的数据
 * @returns 加密/解密后的数据
 */
function processRC4(S: Uint8Array, data: Uint8Array): Uint8Array {
  const result = new Uint8Array(data.length);
  let i = 0;
  let j = 0;
  
  for (let k = 0; k < data.length; k++) {
    i = (i + 1) % 256;
    j = (j + S[i]) % 256;
    
    // 交换 S[i] 和 S[j]
    [S[i], S[j]] = [S[j], S[i]];
    
    // 生成密钥流并进行异或操作
    const keyStreamByte = S[(S[i] + S[j]) % 256];
    result[k] = data[k] ^ keyStreamByte;
  }
  
  return result;
}

/**
 * RC4 加密
 * @param plaintext - 明文字符串
 * @param key - 密钥字符串
 * @returns Base64 编码的密文
 */
export function rc4Encrypt(plaintext: string, key: string): string {
  const S = initRC4(key);
  const plaintextBytes = new TextEncoder().encode(plaintext);
  const cipherBytes = processRC4(S, plaintextBytes);
  
  // 转换为 Base64
  return btoa(String.fromCharCode(...cipherBytes));
}

/**
 * RC4 解密
 * @param ciphertext - Base64 编码的密文
 * @param key - 密钥字符串
 * @returns 明文字符串
 */
export function rc4Decrypt(ciphertext: string, key: string): string {
  const S = initRC4(key);
  
  // 从 Base64 解码
  const cipherBytes = new Uint8Array(
    atob(ciphertext).split('').map(c => c.charCodeAt(0))
  );
  
  const plaintextBytes = processRC4(S, cipherBytes);
  return new TextDecoder().decode(plaintextBytes);
}

/**
 * RC4 加密 (返回十六进制字符串)
 * @param plaintext - 明文字符串
 * @param key - 密钥字符串
 * @returns 十六进制编码的密文
 */
export function rc4EncryptHex(plaintext: string, key: string): string {
  const S = initRC4(key);
  const plaintextBytes = new TextEncoder().encode(plaintext);
  const cipherBytes = processRC4(S, plaintextBytes);
  
  // 转换为十六进制
  return Array.from(cipherBytes)
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * RC4 解密 (从十六进制字符串)
 * @param ciphertext - 十六进制编码的密文
 * @param key - 密钥字符串
 * @returns 明文字符串
 */
export function rc4DecryptHex(ciphertext: string, key: string): string {
  const S = initRC4(key);
  
  // 从十六进制解码
  const cipherBytes = new Uint8Array(
    ciphertext.match(/.{1,2}/g)?.map(byte => Number.parseInt(byte, 16)) || []
  );
  
  const plaintextBytes = processRC4(S, cipherBytes);
  return new TextDecoder().decode(plaintextBytes);
}

/**
 * RC4 加密/解密字节数组
 * @param data - 要处理的字节数组
 * @param key - 密钥字符串或字节数组
 * @returns 处理后的字节数组
 */
export function rc4Bytes(data: Uint8Array, key: string | Uint8Array): Uint8Array {
  const S = initRC4(key);
  return processRC4(S, data);
}

// 默认导出
export default {
  encrypt: rc4Encrypt,
  decrypt: rc4Decrypt,
  encryptHex: rc4EncryptHex,
  decryptHex: rc4DecryptHex,
  bytes: rc4Bytes
};
