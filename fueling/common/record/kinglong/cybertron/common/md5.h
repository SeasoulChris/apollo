/******************************************************************************
 * Copyright 2017 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#ifndef INCLUDE_CYBERTRON_COMMON_MD5_H_
#define INCLUDE_CYBERTRON_COMMON_MD5_H_

#include <stdio.h>
#include <string.h>

// Constants for MD5Transform routine.
const int S11 = 7;
const int S12 = 12;
const int S13 = 17;
const int S14 = 22;
const int S21 = 5;
const int S22 = 9;
const int S23 = 14;
const int S24 = 20;
const int S31 = 4;
const int S32 = 11;
const int S33 = 16;
const int S34 = 23;
const int S41 = 6;
const int S42 = 10;
const int S43 = 15;
const int S44 = 21;

static unsigned char PADDING[64] = {
    0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

// FUN1, FUN2, FUN3 and FUN4 are basic MD5 functions.
#define FUN1(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define FUN2(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define FUN3(x, y, z) ((x) ^ (y) ^ (z))
#define FUN4(x, y, z) ((y) ^ ((x) | (~z)))

// ROTATE_LEFT rotates x left n bits.
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

// TRAN1, TRAN2, TRAN3, and TRAN4 transformations for rounds 1, 2, 3, and 4.
// Rotation is separate from addition to prevent recomputation.
#define TRAN1(a, b, c, d, x, s, ac)                 \
  {                                                 \
    (a) += FUN1((b), (c), (d)) + (x) + (UINT4)(ac); \
    (a) = ROTATE_LEFT((a), (s));                    \
    (a) += (b);                                     \
  }
#define TRAN2(a, b, c, d, x, s, ac)                 \
  {                                                 \
    (a) += FUN2((b), (c), (d)) + (x) + (UINT4)(ac); \
    (a) = ROTATE_LEFT((a), (s));                    \
    (a) += (b);                                     \
  }
#define TRAN3(a, b, c, d, x, s, ac)                 \
  {                                                 \
    (a) += FUN3((b), (c), (d)) + (x) + (UINT4)(ac); \
    (a) = ROTATE_LEFT((a), (s));                    \
    (a) += (b);                                     \
  }
#define TRAN4(a, b, c, d, x, s, ac)                 \
  {                                                 \
    (a) += FUN4((b), (c), (d)) + (x) + (UINT4)(ac); \
    (a) = ROTATE_LEFT((a), (s));                    \
    (a) += (b);                                     \
  }

typedef unsigned char BYTE;

// POINTER defines a generic pointer type
typedef unsigned char* POINTER;

// UINT2 defines a two byte word
typedef unsigned short int UINT2;

// UINT4 defines a four byte word
typedef unsigned int UINT4;

// convenient object that wraps
// the C-functions for use in C++ only
class MD5_cal {
 private:
  struct __context_t {
    UINT4 state[4];           /* state (ABCD) */
    UINT4 count[2];           /* number of bits, modulo 2^64 (lsb first) */
    unsigned char buffer[64]; /* input buffer */
  } context;

  // The core of the MD5 algorithm is here.
  // MD5 basic transformation. Transforms state based on block.
  static void MD5Transform(UINT4 state[4], unsigned char block[64]) {
    UINT4 a = state[0], b = state[1], c = state[2], d = state[3], x[16];
    Decode(x, block, 64);

    /* Round 1 */
    TRAN1(a, b, c, d, x[0], S11, 0xd76aa478);  /* 1 */
    TRAN1(d, a, b, c, x[1], S12, 0xe8c7b756);  /* 2 */
    TRAN1(c, d, a, b, x[2], S13, 0x242070db);  /* 3 */
    TRAN1(b, c, d, a, x[3], S14, 0xc1bdceee);  /* 4 */
    TRAN1(a, b, c, d, x[4], S11, 0xf57c0faf);  /* 5 */
    TRAN1(d, a, b, c, x[5], S12, 0x4787c62a);  /* 6 */
    TRAN1(c, d, a, b, x[6], S13, 0xa8304613);  /* 7 */
    TRAN1(b, c, d, a, x[7], S14, 0xfd469501);  /* 8 */
    TRAN1(a, b, c, d, x[8], S11, 0x698098d8);  /* 9 */
    TRAN1(d, a, b, c, x[9], S12, 0x8b44f7af);  /* 10 */
    TRAN1(c, d, a, b, x[10], S13, 0xffff5bb1); /* 11 */
    TRAN1(b, c, d, a, x[11], S14, 0x895cd7be); /* 12 */
    TRAN1(a, b, c, d, x[12], S11, 0x6b901122); /* 13 */
    TRAN1(d, a, b, c, x[13], S12, 0xfd987193); /* 14 */
    TRAN1(c, d, a, b, x[14], S13, 0xa679438e); /* 15 */
    TRAN1(b, c, d, a, x[15], S14, 0x49b40821); /* 16 */

    /* Round 2 */
    TRAN2(a, b, c, d, x[1], S21, 0xf61e2562);  /* 17 */
    TRAN2(d, a, b, c, x[6], S22, 0xc040b340);  /* 18 */
    TRAN2(c, d, a, b, x[11], S23, 0x265e5a51); /* 19 */
    TRAN2(b, c, d, a, x[0], S24, 0xe9b6c7aa);  /* 20 */
    TRAN2(a, b, c, d, x[5], S21, 0xd62f105d);  /* 21 */
    TRAN2(d, a, b, c, x[10], S22, 0x2441453);  /* 22 */
    TRAN2(c, d, a, b, x[15], S23, 0xd8a1e681); /* 23 */
    TRAN2(b, c, d, a, x[4], S24, 0xe7d3fbc8);  /* 24 */
    TRAN2(a, b, c, d, x[9], S21, 0x21e1cde6);  /* 25 */
    TRAN2(d, a, b, c, x[14], S22, 0xc33707d6); /* 26 */
    TRAN2(c, d, a, b, x[3], S23, 0xf4d50d87);  /* 27 */
    TRAN2(b, c, d, a, x[8], S24, 0x455a14ed);  /* 28 */
    TRAN2(a, b, c, d, x[13], S21, 0xa9e3e905); /* 29 */
    TRAN2(d, a, b, c, x[2], S22, 0xfcefa3f8);  /* 30 */
    TRAN2(c, d, a, b, x[7], S23, 0x676f02d9);  /* 31 */
    TRAN2(b, c, d, a, x[12], S24, 0x8d2a4c8a); /* 32 */

    /* Round 3 */
    TRAN3(a, b, c, d, x[5], S31, 0xfffa3942);  /* 33 */
    TRAN3(d, a, b, c, x[8], S32, 0x8771f681);  /* 34 */
    TRAN3(c, d, a, b, x[11], S33, 0x6d9d6122); /* 35 */
    TRAN3(b, c, d, a, x[14], S34, 0xfde5380c); /* 36 */
    TRAN3(a, b, c, d, x[1], S31, 0xa4beea44);  /* 37 */
    TRAN3(d, a, b, c, x[4], S32, 0x4bdecfa9);  /* 38 */
    TRAN3(c, d, a, b, x[7], S33, 0xf6bb4b60);  /* 39 */
    TRAN3(b, c, d, a, x[10], S34, 0xbebfbc70); /* 40 */
    TRAN3(a, b, c, d, x[13], S31, 0x289b7ec6); /* 41 */
    TRAN3(d, a, b, c, x[0], S32, 0xeaa127fa);  /* 42 */
    TRAN3(c, d, a, b, x[3], S33, 0xd4ef3085);  /* 43 */
    TRAN3(b, c, d, a, x[6], S34, 0x4881d05);   /* 44 */
    TRAN3(a, b, c, d, x[9], S31, 0xd9d4d039);  /* 45 */
    TRAN3(d, a, b, c, x[12], S32, 0xe6db99e5); /* 46 */
    TRAN3(c, d, a, b, x[15], S33, 0x1fa27cf8); /* 47 */
    TRAN3(b, c, d, a, x[2], S34, 0xc4ac5665);  /* 48 */

    /* Round 4 */
    TRAN4(a, b, c, d, x[0], S41, 0xf4292244);  /* 49 */
    TRAN4(d, a, b, c, x[7], S42, 0x432aff97);  /* 50 */
    TRAN4(c, d, a, b, x[14], S43, 0xab9423a7); /* 51 */
    TRAN4(b, c, d, a, x[5], S44, 0xfc93a039);  /* 52 */
    TRAN4(a, b, c, d, x[12], S41, 0x655b59c3); /* 53 */
    TRAN4(d, a, b, c, x[3], S42, 0x8f0ccc92);  /* 54 */
    TRAN4(c, d, a, b, x[10], S43, 0xffeff47d); /* 55 */
    TRAN4(b, c, d, a, x[1], S44, 0x85845dd1);  /* 56 */
    TRAN4(a, b, c, d, x[8], S41, 0x6fa87e4f);  /* 57 */
    TRAN4(d, a, b, c, x[15], S42, 0xfe2ce6e0); /* 58 */
    TRAN4(c, d, a, b, x[6], S43, 0xa3014314);  /* 59 */
    TRAN4(b, c, d, a, x[13], S44, 0x4e0811a1); /* 60 */
    TRAN4(a, b, c, d, x[4], S41, 0xf7537e82);  /* 61 */
    TRAN4(d, a, b, c, x[11], S42, 0xbd3af235); /* 62 */
    TRAN4(c, d, a, b, x[2], S43, 0x2ad7d2bb);  /* 63 */
    TRAN4(b, c, d, a, x[9], S44, 0xeb86d391);  /* 64 */

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;

    // Zeroize sensitive information.
    memset((POINTER)x, 0, sizeof(x));
  }

  // Encodes input (UINT4) into output (unsigned char). Assumes len is
  // a multiple of 4.
  static void Encode(unsigned char* output, UINT4* input, unsigned int len) {
    unsigned int i, j;
    for (i = 0, j = 0; j < len; i++, j += 4) {
      output[j] = (unsigned char)(input[i] & 0xff);
      output[j + 1] = (unsigned char)((input[i] >> 8) & 0xff);
      output[j + 2] = (unsigned char)((input[i] >> 16) & 0xff);
      output[j + 3] = (unsigned char)((input[i] >> 24) & 0xff);
    }
  }

  // Decodes input (unsigned char) into output (UINT4). Assumes len is
  // a multiple of 4.
  static void Decode(UINT4* output, unsigned char* input, unsigned int len) {
    unsigned int i, j;
    for (i = 0, j = 0; j < len; i++, j += 4) {
      output[i] = ((UINT4)input[j]) | (((UINT4)input[j + 1]) << 8) |
                  (((UINT4)input[j + 2]) << 16) | (((UINT4)input[j + 3]) << 24);
    }
  }

 public:
  // MAIN FUNCTIONS
  MD5_cal() { Init(); }

  // MD5 initialization. Begins an MD5 operation, writing a new context.
  void Init() {
    context.count[0] = context.count[1] = 0;

    // Load magic initialization constants.
    context.state[0] = 0x67452301;
    context.state[1] = 0xefcdab89;
    context.state[2] = 0x98badcfe;
    context.state[3] = 0x10325476;
  }

  // MD5 block update operation. Continues an MD5 message-digest
  // operation, processing another message block, and updating the
  // context.
  void Update(unsigned char* input, unsigned int inputLen) {
    unsigned int i, index, partLen;

    // Compute number of bytes mod 64
    index = static_cast<unsigned int>((context.count[0] >> 3) & 0x3F);

    // Update number of bits
    if ((context.count[0] += ((UINT4)inputLen << 3)) < ((UINT4)inputLen << 3)) {
      context.count[1]++;
    }
    context.count[1] += ((UINT4)inputLen >> 29);

    partLen = 64 - index;

    // Transform as many times as possible.
    if (inputLen >= partLen) {
      memcpy((POINTER) & context.buffer[index], (POINTER)input, partLen);
      MD5Transform(context.state, context.buffer);
      for (i = partLen; (i + 63) < inputLen; i += 64) {
        MD5Transform(context.state, &input[i]);
      }
      index = 0;
    } else {
      i = 0;
    }

    /* Buffer remaining input */
    memcpy((POINTER) & context.buffer[index], (POINTER) & input[i],
           inputLen - i);
  }

  // MD5 finalization. Ends an MD5 message-digest operation, writing the
  // the message digest and zeroizing the context.
  // Writes to digestRaw
  void Final() {
    unsigned char bits[8];
    unsigned int index, padLen;

    // Save number of bits
    Encode(bits, context.count, 8);

    // Pad out to 56 mod 64.
    index = static_cast<unsigned int>((context.count[0] >> 3) & 0x3f);
    padLen = (index < 56) ? (56 - index) : (120 - index);
    Update(PADDING, padLen);

    // Append length (before padding)
    Update(bits, 8);

    // Store state in digest
    Encode(digestRaw, context.state, 16);

    // Zeroize sensitive information.
    memset((POINTER) & context, 0, sizeof(context));

    writeToString();
  }

  /// Buffer must be 32+1 (nul) = 33 chars long at least
  void writeToString() {
    int pos;
    for (pos = 0; pos < 16; pos++) {
      sprintf(digestChars + (pos * 2), "%02x", digestRaw[pos]);
    }
  }

 public:
  // an MD5 digest is a 16-byte number (32 hex digits)
  BYTE digestRaw[16];

  // This version of the digest is actually
  // a "printf'd" version of the digest.
  char digestChars[33];

  // Digests a string and prints the result.
  char* digestString(const char* string) {
    Init();
    Update((unsigned char*)string, strlen(string));
    Final();

    return digestChars;
  }
};

#endif
