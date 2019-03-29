// Madgwick filter
// f32 or f64

extern crate quaternion;
extern crate num_traits;

use quaternion as quat;
use quaternion::{Vector3, Quaternion};
use num_traits::float::{Float, FloatConst};


/// IMU(Inertial Measurement Unit)
/// 6軸情報でフィルタ
pub struct Imu<T> {
    beta: T,  // ジャイロセンサの計測誤差によって決まる値
    q: Quaternion<T>,  // 加速度・角速度を用いて推定した結果
}

impl<T> Imu<T> {
    /// gyro_mead_error[deg/s]
    pub fn new(gyro_meas_error: T) -> Imu<T>
    where T: Float + FloatConst {
        let three = T::one() + T::one() + T::one();
        let four  = three + T::one();
        let deg_180 = num_traits::cast(180.0).unwrap();

        // [deg/s] -> [rad/s]
        let gyro_meas_error = T::PI() * (gyro_meas_error / deg_180);
        Imu {
            beta: (three / four).sqrt() * gyro_meas_error,
            q: quat::id::<T>(),
        }
    }

    pub fn update(&mut self, omega: Vector3<T>, accel: Vector3<T>, dt: T) -> Quaternion<T> 
    where T: Float + FloatConst {
        let zero = T::zero();
        let one  = T::one();
        let two  = T::one() + T::one();

        let q0 = self.q.0;
        let q1 = self.q.1[0];
        let q2 = self.q.1[1];
        let q3 = self.q.1[2];

        let half_q = quat::mul_scalar_quat(one / two, self.q);
        let two_q = quat::mul_scalar_quat(two, self.q);

        // Normalize the acceleromeater measurement
        let accel = quat::normalize_vec(accel);
        
        // Compute the objective function and Jacobian
        let f_1 = two * (q1*q3 - q0*q2) - accel[0];
        let f_2 = two * (q0*q1 + q2*q3) - accel[1];
        let f_3 = one - two * (q1*q1 + q2*q2) - accel[2];
        let j_11or24 = two_q.1[1];  // 2 * q2
        let j_12or23 = two_q.1[2];  // 2 * q3
        let j_13or22 = two_q.0;     // 2 * q0
        let j_14or21 = two_q.1[0];  // 2 * q1
        let j_32 = two * j_14or21;
        let j_33 = two * j_11or24;
        
        // Compute gradient (matrix multiplication)
        let tmp0 = j_14or21 * f_2 - j_11or24 * f_1;
        let tmp1 = j_12or23 * f_1 + j_13or22 * f_2 - j_32 * f_3;
        let tmp2 = j_12or23 * f_2 - j_33 * f_3 - j_13or22 * f_1;
        let tmp3 = j_14or21 * f_1 + j_11or24 * f_2;
        let q_hat_dot = quat::normalize( (tmp0, [tmp1, tmp2, tmp3]) );

        // Compute the quaternion derrivative measured by gyroscopes
        let q_dot_omega = quat::mul(half_q, (zero, omega));

        // Compute then integrate the estimated quaternion derrivative
        let tmp0 = quat::mul_scalar_quat(-self.beta, q_hat_dot);
        let tmp1 = quat::add(q_dot_omega, tmp0);
        let tmp = quat::mul_scalar_quat(dt, tmp1);
        self.q = quat::add(self.q, tmp);

        // Normalize quaternion
        self.q = quat::normalize(self.q);

        self.q
    }
}


/// MARG(Magnetic, Angular Rate and Gravity)
/// 9軸情報でフィルタ
pub struct Marg<T> {
    beta: T,
    zeta: T,
    q: Quaternion<T>,
    b_x: T,
    b_z: T,
    omega_bias: [T; 3],  // estimate gyroscope biases error
}

impl<T> Marg<T> {
    /// gyro_meas_error[deg/s]
    /// gyro_meas_drift[deg/s]
    pub fn new(gyro_meas_error: T, gyro_meas_drift: T) -> Marg<T>
    where T: Float + FloatConst {
        let zero = T::zero();
        let one  = T::one();
        let three = one + one + one;
        let four  = three + one;
        let deg_180 = num_traits::cast(180.0).unwrap();

        // [deg/s] -> [rad/s]
        let gyro_meas_error = T::PI() * (gyro_meas_error / deg_180);
        let gyro_meas_drift = T::PI() * (gyro_meas_drift / deg_180);
        Marg {
            beta: (three / four).sqrt() * gyro_meas_error,
            zeta: (three / four).sqrt() * gyro_meas_drift,
            q: quat::id::<T>(),
            b_x: one,
            b_z: zero,
            omega_bias: [zero; 3],
        }
    }

    pub fn update(&mut self, omega: Vector3<T>, accel: Vector3<T>, geomagnetic: Vector3<T>, dt: T) 
    -> Quaternion<T> where T: Float + FloatConst {
        let zero = T::zero();
        let one  = T::one();
        let two  = T::one() + T::one();

        let half_q = quat::mul_scalar_quat(one / two, self.q);
        let two_q = quat::mul_scalar_quat(two, self.q);
        let two_b_x_q = quat::mul_scalar_quat(two * self.b_x, self.q);
        let two_b_z_q = quat::mul_scalar_quat(two * self.b_z, self.q);

        let q0 = self.q.0;
        let q1 = self.q.1[0];
        let q2 = self.q.1[1];
        let q3 = self.q.1[2];

        let mut omega = omega;

        // normalize the accelerometer measurment
        let accel = quat::normalize_vec(accel);

        // normalize the magnetmeter measurement
        let geomagnetic = quat::normalize_vec(geomagnetic);

        // compute the objective function and Jacobian
        let f_1 = two * (q1*q3 - q0*q2) - accel[0];
        let f_2 = two * (q0*q1 + q2*q3) - accel[1];
        let f_3 = one - two * (q1*q1 + q2*q2) - accel[2];
        let f_4 = two * self.b_x * (one/two - q2*q2 - q3*q3) + two * self.b_z * (q1*q3 - q0*q2) - geomagnetic[0];
        let f_5 = two * self.b_x * (q1*q2 - q0*q3) + two * self.b_z * (q0*q1 + q2*q3) - geomagnetic[1];
        let f_6 = two * self.b_x * (q0*q2 + q1*q3) + two * self.b_z * (one/two - q1*q1 - q2*q2) - geomagnetic[2];
        let j_11or24 = two_q.1[1];  // 2 * q2
        let j_12or23 = two_q.1[2];  // 2 * q3
        let j_13or22 = two_q.0;     // 2 * q0
        let j_14or21 = two_q.1[0];  // 2 * q1
        let j_32 = two * j_14or21;
        let j_33 = two * j_11or24;
        let j_41 = two_b_z_q.1[1];  // q2
        let j_42 = two_b_z_q.1[2];  // q3
        let j_43 = two * two_b_x_q.1[1] + two_b_z_q.0;  // 2 * q2 + q0
        let j_44 = two * two_b_x_q.1[2] - two_b_z_q.1[0];  // 2 * q3 - q1
        let j_51 = two_b_x_q.1[2] - two_b_z_q.1[0];  // q3 - q1
        let j_52 = two_b_x_q.1[1] + two_b_z_q.0;  // q2 + q0
        let j_53 = two_b_x_q.1[0] + two_b_z_q.1[2]; // q1 + q3
        let j_54 = two_b_x_q.0 - two_b_z_q.1[1];  // q0 - q2
        let j_61 = two_b_x_q.1[1];  // q2
        let j_62 = two_b_x_q.1[2] - two * two_b_z_q.1[0]; // q3 - 2 * q1
        let j_63 = two_b_x_q.0 - two * two_b_z_q.1[1];  // q0 - 2 * q2
        let j_64 = two_b_x_q.1[0];  // q1

        // compute the gradient (matrix multiplication)
        let tmp0 = j_14or21 * f_2 - j_11or24 * f_1 - j_41 * f_4 - j_51 * f_5 + j_61 * f_6;
        let tmp1 = j_12or23 * f_1 + j_13or22 * f_2 - j_32 * f_3 + j_42 * f_4 + j_52 * f_5 + j_62 * f_6;
        let tmp2 = j_12or23 * f_2 - j_33 * f_3 - j_13or22 * f_1 - j_43 * f_4 + j_53 * f_5 + j_63 * f_6;
        let tmp3 = j_14or21 * f_1 + j_11or24 * f_2 - j_44 * f_4 - j_54 * f_5 + j_64 * f_6;
        // normalize the gradient to estimate direction of the gyroscope error
        let q_hat_dot = quat::normalize( (tmp0, [tmp1, tmp2, tmp3]) );

        // compute angular estimated direction of the gyroscope error
        let omega_error: Vector3<T> = quat::mul(two_q, q_hat_dot).1;

        // compute and remove the gyroscope biases
        for i in 0..3 {
            self.omega_bias[i] = self.omega_bias[i] + omega_error[i] * dt * self.zeta;
            omega[i] = omega[i] - self.omega_bias[i];
        }

        // compute the quaternion rate measured by gyroscopes
        let q_dot_omega = quat::mul(half_q, (zero, omega));

        // compute then integrate the estimated quaternion rate
        let tmp0 = quat::mul_scalar_quat(-self.beta, q_hat_dot);
        let tmp1 = quat::add(q_dot_omega, tmp0);
        let tmp = quat::mul_scalar_quat(dt, tmp1);
        self.q = quat::add(self.q, tmp);

        // normalize quaternion
        self.q = quat::normalize(self.q);

        // compute flux in the earth frame
        let h = quat::vector_rotation(self.q, geomagnetic);

        // normalize the flux vector to have only components in the x and z
        self.b_x = ( h[0]*h[0] + h[1]*h[1] ).sqrt();
        self.b_z = h[3];

        self.q
    }
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
