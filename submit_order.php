<?php
$servername = "localhost";
$username = "root";
$password = "";
$database = "tubelight_db";

$conn = new mysqli($servername, $username, $password, $database);


if ($conn->connect_error) {
  die("Connection failed: " . $conn->connect_error);
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {

  $name    = trim($_POST['name']);
  $email   = trim($_POST['email']);
  $phone   = trim($_POST['phone']);
  $address = trim($_POST['address']);

  $errors = [];

  if (empty($name)) {
    $errors[] = "Name is required.";
  }
  if (empty($email)) {
    $errors[] = "Email is required.";
  } elseif (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
    $errors[] = "Invalid email format.";
  }
  if (empty($phone)) {
    $errors[] = "Phone number is required.";
  }
  if (empty($address)) {
    $errors[] = "Address is required.";
  }

  if (count($errors) > 0) {

    echo "<h2>There were errors with your submission:</h2>";
    echo "<ul>";
    foreach ($errors as $error) {
      echo "<li>" . htmlspecialchars($error) . "</li>";
    }
    echo "</ul>";
    echo "<a href='place-order.html'>Go back to order form</a>";
  } else {
    $sql = "INSERT INTO orders (name, email, phone, address) VALUES (?, ?, ?, ?)";
    $stmt = $conn->prepare($sql);
    $stmt->bind_param("ssss", $name, $email, $phone, $address);

    if ($stmt->execute()) {
      echo "<h2>Thank you, " . htmlspecialchars($name) . "! Your order has been placed successfully.</h2>";
      echo "<a href='index.html'>Return to Home</a>";
    } else {
      echo "Error: " . htmlspecialchars($stmt->error);
    }

    $stmt->close();
  }
} else {

  echo "<h2>Invalid request method.</h2>";
  echo "<a href='place-order.html'>Go back to order form</a>";
}

$conn->close();
?>
