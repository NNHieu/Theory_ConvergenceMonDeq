$$
\begin{aligned}
d\mathbf{z} |_{\mathbf{z}^*} 
=& J\left[ \left( -d A^\top A + dB - dB^\top \right)\mathbf{z}^* + Wd\mathbf{z} + dU \mathbf{x} \right] \\

=& (I - JW)^{-1} J \left[ \left( -d A^\top A + dB - dB^\top \right)\mathbf{z}^* + dU \mathbf{x} \right]
\end{aligned}
$$

$$
\begin{aligned}
\mathbf{u}^\top \mathrm{d}\mathbf{z} |_{\mathbf{z}^*} 
=& \underbrace{\mathbf{u}^\top (I - JW)^{-1} J}_{\mathbf{q}} \left[ \left( -\mathrm{d} A^\top A + \mathrm{d}B - \mathrm{d}B^\top \right)\mathbf{z}^* + \mathrm{d}U \mathbf{x} \right] \\

=& \operatorname{vec}\left\{ 
    \mathbf{q} \left[ \left( -\mathrm{d} A^\top A + \mathrm{d}B - \mathrm{d}B^\top \right)\mathbf{z}^* + \mathrm{d}U \mathbf{x} \right] 
\right\} \\

=& - \operatorname{vec}\left\{ \mathbf{q} \mathrm{d} (A^\top A) \mathbf{z}^*  \right\} \\
&+ \operatorname{vec}\left\{ \mathbf{q} \mathrm{d}\left(B - B^\top \right)\mathbf{z}^* \right\} \\
&+ \operatorname{vec}\left\{ \mathbf{q} \mathrm{d}U \mathbf{x} \right\} \\

=& - \left[ \mathbf{z}^* * \mathbf{q} \right] \mathrm{d} \operatorname{vec}\left( A^\top A \right) \\
&+ \left[ \mathbf{z}^* * \mathbf{q} \right] \mathrm{d} \operatorname{vec}\left( B - B^\top \right)\\
&+ \left[ \mathbf{x} * \mathbf{q} \right] \mathrm{d}\operatorname{vec}\left( U\right) \\

=& - \left[ \mathbf{z}^* * \mathbf{q} \right]  \left[(I_{mm} + K^{(m)})(I\otimes A^T)\right] \mathrm{d} \operatorname{vec} A \\
&+ \left[ \mathbf{z}^* * \mathbf{q} \right]  \left( I_{mm} - K^{(m,m)} \right)\mathrm{d} \operatorname{vec} B\\
&+ \left[ \mathbf{x} * \mathbf{q} \right] \mathrm{d}\operatorname{vec} U \\

=& - \left[ \mathbf{z}^* * \mathbf{q} \right]  P_A \mathrm{d} \operatorname{vec} A \\
&+ \left[ \mathbf{z}^* * \mathbf{q} \right]  P_B \mathrm{d} \operatorname{vec} B\\
&+ \left[ \mathbf{x} * \mathbf{q} \right] \mathrm{d}\operatorname{vec} U \\

\end{aligned}
$$

$$
\begin{aligned}
\mathbf{u}^\top \mathrm{d}Z |_{Z^*}
=& - \left[ Z^* * Q \right]  P_A \mathrm{d} \operatorname{vec} A \\
&+ \left[ Z^* * Q \right]  P_B \mathrm{d} \operatorname{vec} B\\
&+ \left[ X * Q \right] \mathrm{d}\operatorname{vec} U \\ 
\end{aligned}
$$

$$
\begin{aligned}
\mathrm{d} \mathbf{y} = \mathbf{u}^\top \mathrm{d} Z|_{Z^*} + \mathrm{d} \mathbf{u}^\top Z^*
\end{aligned}
$$

$$
\begin{aligned}
H 
=& \left[ \frac{\partial \mathbf{y}}{\partial\operatorname{vec} A} \right] \left[ \frac{\partial \mathbf{y}}{\partial\operatorname{vec} A} \right]^T
+ \left[ \frac{\partial \mathbf{y}}{\partial\operatorname{vec} B} \right] \left[ \frac{\partial \mathbf{y}}{\partial\operatorname{vec} B} \right]^T
+ \left[ \frac{\partial \mathbf{y}}{\partial\operatorname{vec} U} \right] \left[ \frac{\partial \mathbf{y}}{\partial\operatorname{vec} U} \right]^T
+ \left[ \frac{\partial \mathbf{y}}{\partial \mathbf{u}} \right] \left[ \frac{\partial \mathbf{y}}{\partial \mathbf{u}} \right]^T \\
=& \left[ Z^* * Q \right] \left(-P_A P_A^T + P_B P_B^T \right) \left[ Z^* * Q \right]^T + \left[ X * Q \right] \left[ X * Q \right]^T + Z^*(Z^*)^T
\end{aligned}
$$

$$
\left[ X * Q \right] \left[ X * Q \right]^T = \left[ X X^T \right]\circ \left[ QQ^T \right]
$$

$$
\begin{aligned}    
P_A P_A^T     =& \left[(I + K^{(m)})(I \otimes A^T)\right] \left[(I + K^{(m)})(I \otimes A^T)\right]^\top \\    
=& (I + K^{(m)})(I \otimes A^T) (I \otimes A^T)^\top (I + K^{(m)})^\top \\    
=& (I + K^{(m)})(I \otimes A^T) (I \otimes A)(I + K^{(m)}) \\    
=& (I + K^{(m)}) (I \otimes A^\top A)(I + K^{(m)}) \\
=& (I \otimes A^\top A) + K^{(m)}(I \otimes A^\top A) \\
&+ K^{(m)}(A^\top A \otimes I) + (A^\top A \otimes I) \\
=& (I + K^{(m)})(A^TA \oplus A^TA)
\end{aligned}
$$