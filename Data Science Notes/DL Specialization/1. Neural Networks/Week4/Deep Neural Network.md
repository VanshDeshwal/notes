Cant predict how many layers will be best for a given problem so we try different values


$n^{[l]}$ = # units in layer l
$a^{[l]}$ = activations in layer l

## Forward Propagation in a Deep Network
For single training sample $x$

$z^{[1]} = W^{[1]}x + b^{[1]}$
$a^{[1]} = g^{[1]}(z^{[1]})$

$z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}$
$a^{[2]} = g^{[2]}(z^{[2]})$

$z^{[3]} = W^{[3]}a^{[2]} + b^{[3]}$
$a^{[3]} = g^{[3]}(z^{[3]})$

$z^{[4]} = W^{[4]}a^{[3]} + b^{[4]}$
$a^{[4]} = g^{[4]}(z^{[4]})$

$z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$
$a^{[l]} = g^{[l]}(z^{[l]})$

### Vectorizing for m samples

$X^{[1]} = W^{[1]}X + b^{[1]}$
$A^{[1]} = g^{[1]}(Z^{[1]})$

$Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}$
$A^{[2]} = g^{[2]}(Z^{[2]})$

$Z^{[3]} = W^{[3]}A^{[2]} + b^{[3]}$
$A^{[3]} = g^{[3]}(Z^{[3]})$

$Z^{[4]} = W^{[4]}A^{[3]} + b^{[4]}$
$A^{[4]} = g^{[4]}(Z^{[4]})$

$Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}$
$A^{[l]} = g^{[l]}(Z^{[l]})$

we have to calculate these for each layer using a loop, cant apply vectorization here

