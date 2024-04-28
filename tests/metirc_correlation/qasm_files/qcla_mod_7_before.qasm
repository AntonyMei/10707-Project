OPENQASM 2.0;
include "qelib1.inc";
qreg q[26];
h q[0];
h q[3];
h q[4];
h q[7];
h q[8];
h q[11];
h q[12];
h q[15];
h q[18];
h q[19];
h q[22];
h q[23];
cx q[2], q[4];
tdg q[4];
cx q[1], q[4];
t q[4];
 cx q[2], q[4];
tdg q[4];
 cx q[1], q[4];
 cx q[1], q[2];
tdg q[2];
 cx q[1], q[2];
t q[1];
 t q[2];
 t q[4];
cx q[6], q[8];
tdg q[8];
cx q[5], q[8];
t q[8];
 cx q[6], q[8];
tdg q[8];
 cx q[5], q[8];
 cx q[5], q[6];
tdg q[6];
 cx q[5], q[6];
t q[5];
 t q[6];
 t q[8];
cx q[10], q[12];
tdg q[12];
cx q[9], q[12];
t q[12];
 cx q[10], q[12];
tdg q[12];
 cx q[9], q[12];
 cx q[9], q[10];
tdg q[10];
 cx q[9], q[10];
t q[9];
 t q[10];
 t q[12];
cx q[14], q[15];
tdg q[15];
cx q[13], q[15];
t q[15];
 cx q[14], q[15];
tdg q[15];
 cx q[13], q[15];
 cx q[13], q[14];
tdg q[14];
 cx q[13], q[14];
t q[13];
 t q[14];
 t q[15];
cx q[17], q[19];
tdg q[19];
cx q[16], q[19];
t q[19];
 cx q[17], q[19];
tdg q[19];
 cx q[16], q[19];
 cx q[16], q[17];
tdg q[17];
 cx q[16], q[17];
t q[16];
 t q[17];
 t q[19];
cx q[21], q[23];
tdg q[23];
cx q[20], q[23];
t q[23];
 cx q[21], q[23];
tdg q[23];
 cx q[20], q[23];
 cx q[20], q[21];
tdg q[21];
 cx q[20], q[21];
t q[20];
 t q[21];
 t q[23];
cx q[25], q[0];
tdg q[0];
cx q[24], q[0];
t q[0];
 cx q[25], q[0];
tdg q[0];
 cx q[24], q[0];
 cx q[24], q[25];
tdg q[25];
 cx q[24], q[25];
t q[24];
 t q[25];
 t q[0];
cx q[1], q[2];
cx q[5], q[6];
cx q[9], q[10];
cx q[13], q[14];
cx q[16], q[17];
cx q[20], q[21];
cx q[24], q[25];
cx q[6], q[3];
tdg q[3];
cx q[2], q[3];
t q[3];
 cx q[6], q[3];
tdg q[3];
 cx q[2], q[3];
 cx q[2], q[6];
tdg q[6];
 cx q[2], q[6];
t q[2];
 t q[6];
 t q[3];
cx q[14], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
 cx q[14], q[11];
tdg q[11];
 cx q[10], q[11];
 cx q[10], q[14];
tdg q[14];
 cx q[10], q[14];
t q[10];
 t q[14];
 t q[11];
cx q[21], q[18];
tdg q[18];
cx q[17], q[18];
t q[18];
 cx q[21], q[18];
tdg q[18];
 cx q[17], q[18];
 cx q[17], q[21];
tdg q[21];
 cx q[17], q[21];
t q[17];
 t q[21];
 t q[18];
h q[3];
h q[4];
h q[11];
h q[12];
h q[18];
h q[19];
cx q[11], q[7];
tdg q[7];
cx q[3], q[7];
t q[7];
 cx q[11], q[7];
tdg q[7];
 cx q[3], q[7];
 cx q[3], q[11];
tdg q[11];
 cx q[3], q[11];
t q[3];
 t q[11];
 t q[7];
cx q[25], q[22];
tdg q[22];
cx q[18], q[22];
t q[22];
 cx q[25], q[22];
tdg q[22];
 cx q[18], q[22];
 cx q[18], q[25];
tdg q[25];
 cx q[18], q[25];
t q[18];
 t q[25];
 t q[22];
cx q[6], q[8];
tdg q[8];
cx q[4], q[8];
t q[8];
 cx q[6], q[8];
tdg q[8];
 cx q[4], q[8];
 cx q[4], q[6];
tdg q[6];
 cx q[4], q[6];
t q[4];
 t q[6];
 t q[8];
cx q[14], q[15];
tdg q[15];
cx q[12], q[15];
t q[15];
 cx q[14], q[15];
tdg q[15];
 cx q[12], q[15];
 cx q[12], q[14];
tdg q[14];
 cx q[12], q[14];
t q[12];
 t q[14];
 t q[15];
cx q[21], q[23];
tdg q[23];
cx q[19], q[23];
t q[23];
 cx q[21], q[23];
tdg q[23];
 cx q[19], q[23];
 cx q[19], q[21];
tdg q[21];
 cx q[19], q[21];
t q[19];
 t q[21];
 t q[23];
h q[7];
h q[8];
h q[22];
h q[23];
cx q[11], q[15];
tdg q[15];
cx q[8], q[15];
t q[15];
 cx q[11], q[15];
tdg q[15];
 cx q[8], q[15];
 cx q[8], q[11];
tdg q[11];
 cx q[8], q[11];
t q[8];
 t q[11];
 t q[15];
h q[15];
cx q[22], q[0];
tdg q[0];
cx q[7], q[0];
t q[0];
 cx q[22], q[0];
tdg q[0];
 cx q[7], q[0];
 cx q[7], q[22];
tdg q[22];
 cx q[7], q[22];
t q[7];
 t q[22];
 t q[0];
cx q[23], q[0];
tdg q[0];
cx q[25], q[0];
t q[0];
 cx q[23], q[0];
tdg q[0];
 cx q[25], q[0];
 cx q[25], q[23];
tdg q[23];
 cx q[25], q[23];
t q[25];
 t q[23];
 t q[0];
cx q[15], q[0];
tdg q[0];
cx q[22], q[0];
t q[0];
 cx q[15], q[0];
tdg q[0];
 cx q[22], q[0];
 cx q[22], q[15];
tdg q[15];
 cx q[22], q[15];
t q[22];
 t q[15];
 t q[0];
h q[0];
h q[15];
cx q[7], q[15];
tdg q[15];
cx q[0], q[15];
t q[15];
 cx q[7], q[15];
tdg q[15];
 cx q[0], q[15];
 cx q[0], q[7];
tdg q[7];
 cx q[0], q[7];
t q[0];
 t q[7];
 t q[15];
h q[15];
h q[8];
h q[7];
h q[4];
h q[19];
h q[22];
h q[23];
cx q[3], q[8];
tdg q[8];
cx q[0], q[8];
t q[8];
 cx q[3], q[8];
tdg q[8];
 cx q[0], q[8];
 cx q[0], q[3];
tdg q[3];
 cx q[0], q[3];
t q[0];
 t q[3];
 t q[8];
cx q[2], q[4];
tdg q[4];
cx q[0], q[4];
t q[4];
 cx q[2], q[4];
tdg q[4];
 cx q[0], q[4];
 cx q[0], q[2];
tdg q[2];
 cx q[0], q[2];
t q[0];
 t q[2];
 t q[4];
cx q[11], q[7];
tdg q[7];
cx q[3], q[7];
t q[7];
 cx q[11], q[7];
tdg q[7];
 cx q[3], q[7];
 cx q[3], q[11];
tdg q[11];
 cx q[3], q[11];
t q[3];
 t q[11];
 t q[7];
cx q[18], q[23];
tdg q[23];
cx q[15], q[23];
t q[23];
 cx q[18], q[23];
tdg q[23];
 cx q[15], q[23];
 cx q[15], q[18];
tdg q[18];
 cx q[15], q[18];
t q[15];
 t q[18];
 t q[23];
cx q[17], q[19];
tdg q[19];
cx q[15], q[19];
t q[19];
 cx q[17], q[19];
tdg q[19];
 cx q[15], q[19];
 cx q[15], q[17];
tdg q[17];
 cx q[15], q[17];
t q[15];
 t q[17];
 t q[19];
cx q[25], q[22];
tdg q[22];
cx q[18], q[22];
t q[22];
 cx q[25], q[22];
tdg q[22];
 cx q[18], q[22];
 cx q[18], q[25];
tdg q[25];
 cx q[18], q[25];
t q[18];
 t q[25];
 t q[22];
h q[4];
h q[8];
h q[19];
h q[23];
h q[3];
h q[11];
h q[12];
h q[18];
cx q[6], q[3];
tdg q[3];
cx q[2], q[3];
t q[3];
 cx q[6], q[3];
tdg q[3];
 cx q[2], q[3];
 cx q[2], q[6];
tdg q[6];
 cx q[2], q[6];
t q[2];
 t q[6];
 t q[3];
cx q[14], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
 cx q[14], q[11];
tdg q[11];
 cx q[10], q[11];
 cx q[10], q[14];
tdg q[14];
 cx q[10], q[14];
t q[10];
 t q[14];
 t q[11];
cx q[10], q[12];
tdg q[12];
cx q[8], q[12];
t q[12];
 cx q[10], q[12];
tdg q[12];
 cx q[8], q[12];
 cx q[8], q[10];
tdg q[10];
 cx q[8], q[10];
t q[8];
 t q[10];
 t q[12];
cx q[21], q[18];
tdg q[18];
cx q[17], q[18];
t q[18];
 cx q[21], q[18];
tdg q[18];
 cx q[17], q[18];
 cx q[17], q[21];
tdg q[21];
 cx q[17], q[21];
t q[17];
 t q[21];
 t q[18];
h q[12];
cx q[0], q[2];
cx q[4], q[6];
cx q[8], q[10];
cx q[12], q[14];
cx q[15], q[17];
cx q[19], q[21];
cx q[23], q[25];
x q[2];
x q[6];
x q[10];
x q[14];
x q[17];
x q[21];
x q[25];
cx q[1], q[2];
cx q[5], q[6];
cx q[9], q[10];
cx q[13], q[14];
cx q[16], q[17];
cx q[20], q[21];
cx q[24], q[25];
h q[12];
cx q[6], q[3];
tdg q[3];
cx q[2], q[3];
t q[3];
 cx q[6], q[3];
tdg q[3];
 cx q[2], q[3];
 cx q[2], q[6];
tdg q[6];
 cx q[2], q[6];
t q[2];
 t q[6];
 t q[3];
cx q[14], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
 cx q[14], q[11];
tdg q[11];
 cx q[10], q[11];
 cx q[10], q[14];
tdg q[14];
 cx q[10], q[14];
t q[10];
 t q[14];
 t q[11];
cx q[10], q[12];
tdg q[12];
cx q[8], q[12];
t q[12];
 cx q[10], q[12];
tdg q[12];
 cx q[8], q[12];
 cx q[8], q[10];
tdg q[10];
 cx q[8], q[10];
t q[8];
 t q[10];
 t q[12];
cx q[21], q[18];
tdg q[18];
cx q[17], q[18];
t q[18];
 cx q[21], q[18];
tdg q[18];
 cx q[17], q[18];
 cx q[17], q[21];
tdg q[21];
 cx q[17], q[21];
t q[17];
 t q[21];
 t q[18];
h q[3];
h q[11];
h q[12];
h q[18];
h q[4];
h q[8];
h q[19];
h q[23];
cx q[3], q[8];
tdg q[8];
cx q[0], q[8];
t q[8];
 cx q[3], q[8];
tdg q[8];
 cx q[0], q[8];
 cx q[0], q[3];
tdg q[3];
 cx q[0], q[3];
t q[0];
 t q[3];
 t q[8];
cx q[2], q[4];
tdg q[4];
cx q[0], q[4];
t q[4];
 cx q[2], q[4];
tdg q[4];
 cx q[0], q[4];
 cx q[0], q[2];
tdg q[2];
 cx q[0], q[2];
t q[0];
 t q[2];
 t q[4];
cx q[11], q[7];
tdg q[7];
cx q[3], q[7];
t q[7];
 cx q[11], q[7];
tdg q[7];
 cx q[3], q[7];
 cx q[3], q[11];
tdg q[11];
 cx q[3], q[11];
t q[3];
 t q[11];
 t q[7];
cx q[18], q[23];
tdg q[23];
cx q[15], q[23];
t q[23];
 cx q[18], q[23];
tdg q[23];
 cx q[15], q[23];
 cx q[15], q[18];
tdg q[18];
 cx q[15], q[18];
t q[15];
 t q[18];
 t q[23];
cx q[17], q[19];
tdg q[19];
cx q[15], q[19];
t q[19];
 cx q[17], q[19];
tdg q[19];
 cx q[15], q[19];
 cx q[15], q[17];
tdg q[17];
 cx q[15], q[17];
t q[15];
 t q[17];
 t q[19];
cx q[25], q[22];
tdg q[22];
cx q[18], q[22];
t q[22];
 cx q[25], q[22];
tdg q[22];
 cx q[18], q[22];
 cx q[18], q[25];
tdg q[25];
 cx q[18], q[25];
t q[18];
 t q[25];
 t q[22];
h q[8];
h q[7];
h q[4];
h q[19];
h q[22];
h q[23];
h q[15];
cx q[7], q[15];
tdg q[15];
cx q[0], q[15];
t q[15];
 cx q[7], q[15];
tdg q[15];
 cx q[0], q[15];
 cx q[0], q[7];
tdg q[7];
 cx q[0], q[7];
t q[0];
 t q[7];
 t q[15];
h q[0];
h q[15];
cx q[23], q[0];
tdg q[0];
cx q[25], q[0];
t q[0];
 cx q[23], q[0];
tdg q[0];
 cx q[25], q[0];
 cx q[25], q[23];
tdg q[23];
 cx q[25], q[23];
t q[25];
 t q[23];
 t q[0];
cx q[15], q[0];
tdg q[0];
cx q[22], q[0];
t q[0];
 cx q[15], q[0];
tdg q[0];
 cx q[22], q[0];
 cx q[22], q[15];
tdg q[15];
 cx q[22], q[15];
t q[22];
 t q[15];
 t q[0];
h q[15];
cx q[11], q[15];
tdg q[15];
cx q[8], q[15];
t q[15];
 cx q[11], q[15];
tdg q[15];
 cx q[8], q[15];
 cx q[8], q[11];
tdg q[11];
 cx q[8], q[11];
t q[8];
 t q[11];
 t q[15];
h q[7];
h q[8];
h q[22];
h q[23];
cx q[11], q[7];
tdg q[7];
cx q[3], q[7];
t q[7];
 cx q[11], q[7];
tdg q[7];
 cx q[3], q[7];
 cx q[3], q[11];
tdg q[11];
 cx q[3], q[11];
t q[3];
 t q[11];
 t q[7];
cx q[25], q[22];
tdg q[22];
cx q[18], q[22];
t q[22];
 cx q[25], q[22];
tdg q[22];
 cx q[18], q[22];
 cx q[18], q[25];
tdg q[25];
 cx q[18], q[25];
t q[18];
 t q[25];
 t q[22];
cx q[6], q[8];
tdg q[8];
cx q[4], q[8];
t q[8];
 cx q[6], q[8];
tdg q[8];
 cx q[4], q[8];
 cx q[4], q[6];
tdg q[6];
 cx q[4], q[6];
t q[4];
 t q[6];
 t q[8];
cx q[14], q[15];
tdg q[15];
cx q[12], q[15];
t q[15];
 cx q[14], q[15];
tdg q[15];
 cx q[12], q[15];
 cx q[12], q[14];
tdg q[14];
 cx q[12], q[14];
t q[12];
 t q[14];
 t q[15];
cx q[21], q[23];
tdg q[23];
cx q[19], q[23];
t q[23];
 cx q[21], q[23];
tdg q[23];
 cx q[19], q[23];
 cx q[19], q[21];
tdg q[21];
 cx q[19], q[21];
t q[19];
 t q[21];
 t q[23];
h q[3];
h q[4];
h q[11];
h q[12];
h q[18];
h q[19];
cx q[6], q[3];
tdg q[3];
cx q[2], q[3];
t q[3];
 cx q[6], q[3];
tdg q[3];
 cx q[2], q[3];
 cx q[2], q[6];
tdg q[6];
 cx q[2], q[6];
t q[2];
 t q[6];
 t q[3];
cx q[14], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
 cx q[14], q[11];
tdg q[11];
 cx q[10], q[11];
 cx q[10], q[14];
tdg q[14];
 cx q[10], q[14];
t q[10];
 t q[14];
 t q[11];
cx q[21], q[18];
tdg q[18];
cx q[17], q[18];
t q[18];
 cx q[21], q[18];
tdg q[18];
 cx q[17], q[18];
 cx q[17], q[21];
tdg q[21];
 cx q[17], q[21];
t q[17];
 t q[21];
 t q[18];
cx q[1], q[2];
cx q[5], q[6];
cx q[9], q[10];
cx q[13], q[14];
cx q[16], q[17];
cx q[20], q[21];
cx q[24], q[25];
cx q[2], q[4];
tdg q[4];
cx q[1], q[4];
t q[4];
 cx q[2], q[4];
tdg q[4];
 cx q[1], q[4];
 cx q[1], q[2];
tdg q[2];
 cx q[1], q[2];
t q[1];
 t q[2];
 t q[4];
cx q[6], q[8];
tdg q[8];
cx q[5], q[8];
t q[8];
 cx q[6], q[8];
tdg q[8];
 cx q[5], q[8];
 cx q[5], q[6];
tdg q[6];
 cx q[5], q[6];
t q[5];
 t q[6];
 t q[8];
cx q[10], q[12];
tdg q[12];
cx q[9], q[12];
t q[12];
 cx q[10], q[12];
tdg q[12];
 cx q[9], q[12];
 cx q[9], q[10];
tdg q[10];
 cx q[9], q[10];
t q[9];
 t q[10];
 t q[12];
cx q[14], q[15];
tdg q[15];
cx q[13], q[15];
t q[15];
 cx q[14], q[15];
tdg q[15];
 cx q[13], q[15];
 cx q[13], q[14];
tdg q[14];
 cx q[13], q[14];
t q[13];
 t q[14];
 t q[15];
cx q[17], q[19];
tdg q[19];
cx q[16], q[19];
t q[19];
 cx q[17], q[19];
tdg q[19];
 cx q[16], q[19];
 cx q[16], q[17];
tdg q[17];
 cx q[16], q[17];
t q[16];
 t q[17];
 t q[19];
cx q[21], q[23];
tdg q[23];
cx q[20], q[23];
t q[23];
 cx q[21], q[23];
tdg q[23];
 cx q[20], q[23];
 cx q[20], q[21];
tdg q[21];
 cx q[20], q[21];
t q[20];
 t q[21];
 t q[23];
cx q[25], q[0];
tdg q[0];
cx q[24], q[0];
t q[0];
 cx q[25], q[0];
tdg q[0];
 cx q[24], q[0];
 cx q[24], q[25];
tdg q[25];
 cx q[24], q[25];
t q[24];
 t q[25];
 t q[0];
h q[0];
h q[3];
h q[4];
h q[7];
h q[8];
h q[11];
h q[12];
h q[15];
h q[18];
h q[19];
h q[22];
h q[23];
