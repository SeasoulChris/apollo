<!doctype html>
<html lang="en">

<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Apollo Job Submission</title>

    <!-- Bootstrap core CSS -->
    <link href="https://getbootstrap.com/docs/4.3/dist/css/bootstrap.min.css" rel="stylesheet"
        integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">

    <style>
        .bd-placeholder-img {
            font-size: 1.125rem;
            text-anchor: middle;
            -webkit-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none;
            user-select: none;
        }

        @media (min-width: 768px) {
            .bd-placeholder-img-lg {
                font-size: 3.5rem;
            }
        }

        .container {
            max-width: 960px;
        }

        .lh-condensed {
            line-height: 1.25;
        }
    </style>
</head>

<body class="bg-light">

    <div class="container">
        <nav class="navbar navbar-expand-lg navbar-light bg-light">
            <a class="navbar-brand" href="#">Apollo Job form submission</a>
        </nav>

        <div class="row">

            <div class="col-md-8 order-md-1">
                <h4 class="mb-3">Partner Info</h4>
                <form class="needs-validation" novalidate>
                    <div class="mb-3">
                        <label for="partner_id">Partner id </label>
                        <div class="input-group">
                            <input type="text" class="form-control" id="partner_id" placeholder="Job Owner" required>
                            <div class="invalid-feedback" style="width: 100%;">
                                Partner id is required.
                            </div>
                        </div>
                    </div>

                    <div class="mb-3">
                        <label for="email">Email <span class="text-muted">to accept job report</span></label>
                        <input type="email" class="form-control" id="email" placeholder="you@example.com">
                        <div class="invalid-feedback">
                            Please enter a valid email address for acceptting the job report.
                        </div>
                    </div>

                    <hr class="mb-4">

                    <h4 class="mb-3">Job</h4>

                    <div class=" mb-3">
                        <label for="job">Job Type</label>
                        <select class="custom-select d-block w-100" id="job_type" required>
                            <option value="">Choose...</option>
                            <option value="VEHICLE_CALIBRATION">VEHICLE_CALIBRATION</option>
                            <option value="SIMPLE_HDMAP">SIMPLE_HDMAP</option>
                            <!-- <option value="">Sensor Calibration </option> -->
                            <!-- <option>Visualization Model Training</option> -->
                        </select>
                        <div class="invalid-feedback">
                            Please select a Job Type.
                        </div>
                    </div>


                    <hr class="mb-4">

                    <h4 class="mb-3">Storage</h4>

                    <div class="d-block my-3">
                        <div class="custom-control custom-radio">
                            <input id="bos" type="radio" class="custom-control-input" checked required>
                            <label class="custom-control-label" for="bos">BOS</label>
                        </div>
                        <div class="custom-control custom-radio">
                            <input id="blob" type="radio" class="custom-control-input" required>
                            <label class="custom-control-label" for="blob">AZURE Blob</label>
                        </div>
                    </div>

                    <div class=" mb-3">
                        <label for="zip">Bucket Name</label>
                        <input type="text" class="form-control" id="bucket" placeholder="" required>
                        <div class="invalid-feedback">
                            Bucket Name required.
                        </div>
                    </div>

                    <div class="row">
                        <div class="col-md-6 mb-3">
                            <label for="cc-name">Access Key</label>
                            <input type="text" class="form-control" id="access_key" placeholder="" required>
                            <div class="invalid-feedback">
                                worker count is required
                            </div>
                        </div>
                        <div class="col-md-6 mb-3">
                            <label for="cc-number">Access Secret</label>
                            <input type="text" class="form-control" id="access_secret" placeholder="" required>
                            <div class="invalid-feedback">
                                lidar_type is required
                            </div>
                        </div>
                    </div>

                    <div class=" mb-3">
                        <label for="input_data_path">Input Data Path</label>
                        <input type="text" class="form-control" id="input_data_path" placeholder="" required>
                        <div class="invalid-feedback">
                            Input Data Path required.
                        </div>
                    </div>

                    <hr class="mb-4">

                    <div class="custom-control custom-checkbox">
                        <input type="checkbox" class="custom-control-input" id="writable">
                        <label class="custom-control-label" for="writable">Storage Writable ?</label>
                    </div>

                    <hr class="mb-4">

                    <div class="row">
                        <div class="col-md-6 mb-3">
                            <label for="cc-name">Zone Id</label>
                            <input type="text" class="form-control" id="zone_id" placeholder="">
                            <div class="invalid-feedback">
                                Zone Id is required
                            </div>
                        </div>
                        <div class="col-md-6 mb-3">
                            <label for="cc-number">Lidar type</label>
                            <input type="text" class="form-control" id="lidar_type" placeholder="">
                            <div class="invalid-feedback">
                                Lidar type is required
                            </div>
                        </div>
                    </div>

                    <hr class="mb-4">
                    <button class="btn btn-primary btn-lg btn-block" type="submit">Continue to
                        checkout</button>
                </form>
            </div>
        </div>

        <footer class="my-5 pt-5 text-muted text-center text-small">
            <p class="mb-1">&copy; 2019 Baidu</p>
        </footer>
    </div>
    <script src="https://code.jquery.com/jquery-3.1.1.min.js"></script>
    <script src="https://getbootstrap.com/docs/4.3/dist/js/bootstrap.bundle.min.js"
        integrity="sha384-xrRywqdh3PHs8keKZN+8zzc5TX0GRTLCcmivcbNJWm2rs5C8PRhcEn3czEjhAO9o"
        crossorigin="anonymous"></script>
    <script>
        // Example starter JavaScript for disabling form submissions if there are invalid fields
        (function () {
            'use strict'

            window.addEventListener('load', function () {
                // Fetch all the forms we want to apply custom Bootstrap validation styles to
                var forms = document.getElementsByClassName('needs-validation')

                // Loop over them and prevent submission
                Array.prototype.filter.call(forms, function (form) {
                    form.addEventListener('submit', function (event) {
                        if (form.checkValidity() === false) {
                            event.preventDefault()
                            event.stopPropagation()
                        } else {
                            submitForm();
                        }
                        form.classList.add('was-validated')
                    }, false)
                })
            }, false)
        }())

        function submitForm() {
            // Initiate Variables With Form Content
            var partner_id = $("#partner_id").val();
            var email = $("#email").val();
            var job_type = $("#job_type").val();
            var bucket = $("#bucket").val();
            var access_key = $("#access_key").val();
            var access_secret = $("#access_secret").val();
            var input_data_path = $("#input_data_path").val();
            var bos = $("#bos").is(':checked');
            var blob = $("#blob").is(':checked');
            var partner_storage_writable = $("#writable").is(':checked');
            var zone_id = $("#zone_id").val();
            var lidar_type = $("#lidar_type").val();

            var data = {
                "partner_id": partner_id,
                "job_type": job_type,
                "bos": bos,
                "blob": blob,
                "bucket":bucket,
                "access_key":access_key,
                "access_secret":access_secret,
                "input_data_path": input_data_path,
                "partner_storage_writable":partner_storage_writable,
                "zone_id":zone_id,
                "lidar_type":lidar_type
            }

            $.ajax({
                type: "POST",
                url: "http://localhost:5000/submit_job",
                data: JSON.stringify(data),
                contentType: 'application/json;charset=UTF-8',
                success: function (response) {
                    console.log(response);
                    window.alert(response.message);
                    window.location = "http://localhost:5000"
                }
            });
        }

    </script>
</body>

</html>
